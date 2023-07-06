from plum import dispatch
from typing import Optional, Literal, Callable
from dataclasses import dataclass

import dgl
import dgl.function as fn
from dgl.nn import AvgPooling, SumPooling
from dgl.nn.functional import edge_softmax

import torch
from torch.nn import functional as F
from torch import nn

from nfflr.models.utils import (
    autograd_forces,
    RBFExpansion,
    ChebyshevExpansion,
    MLPLayer,
)

from nfflr.nn.norm import Norm
from nfflr.nn.transform import PeriodicRadiusGraph
from nfflr.nn.cutoff import XPLOR
from nfflr.data.atoms import _get_attribute_lookup, Atoms


def edge_attention_graph(g: dgl.DGLGraph, shared=False):

    # get all pairs of incident edges for each node
    # torch.combinations gives half the pairs
    eids = torch.vstack(
        [torch.combinations(g.in_edges(id_node, form="eid")) for id_node in g.nodes()]
    )

    src, dst = eids.T

    # to_bidirected fills in the other half of edge pairs
    # all at once
    t = dgl.to_bidirected(dgl.graph((src, dst)))

    if shared:
        t.ndata["r"] = g.edata["r"]

    return t


@dataclass
class TFMConfig:
    """Hyperparameter schema for nfflr.models.gnn.tfm"""

    transform: Callable = PeriodicRadiusGraph(cutoff=8.0)
    cutoff: torch.nn.Module = XPLOR(7.5, 8.0)
    layers: int = 3
    norm: Literal["layernorm", "instancenorm"] = "layernorm"
    atom_features: str = "embedding"
    d_model: int = 256
    d_message: int = 64
    output_features: int = 1
    compute_forces: bool = False
    energy_units: Literal["eV", "eV/atom"] = "eV/atom"
    reference_energies: Optional[torch.Tensor] = None


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also https://www.jmlr.org/papers/v24/22-0567.html


    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        norm: Literal["layernorm", "instancenorm", "batchnorm"] = "layernorm",
    ):
        """Edge gated graph convolution variant."""
        super().__init__()

        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.norm_edges = Norm(output_features, norm_type=norm, mode="edge")

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.norm_nodes = Norm(output_features, norm_type=norm, mode="node")

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))

        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))

        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)
        # Dwivedi et al apply the nonlinearity and edge residual
        # before the attention process
        y = edge_feats + F.silu(self.norm_edges(g, m))

        # if edge attributes have a cutoff function value
        # multiply the edge gate values with the cutoff value
        cutoff_value = g.edata.get("cutoff_value")
        if cutoff_value is not None:
            g.edata["sigma"] = torch.sigmoid(y) * cutoff_value.unsqueeze(1)
        else:
            g.edata["sigma"] = torch.sigmoid(y)

        # compute pair interaction modulated by edge gates
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h"))
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        x = node_feats + F.silu(self.norm_nodes(g, x))

        return x, y


class FeedForward(nn.Module):
    """feedforward layer."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        """Linear SiLU Linear feedforward layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        """Linear, norm, silu layer."""
        return self.layer(x)


class TersoffAttention(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.channels = channels
        self.angle_encoder = ChebyshevExpansion(channels)
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, channels)))

    def forward(self, g: dgl.DGLGraph, t: dgl.DGLGraph, xij: torch.Tensor):

        # Tersoff attention
        t.ndata["xij"] = xij

        # compute bond angle embedding
        t.ndata["rnorm"] = -t.ndata["r"] / torch.norm(t.ndata["r"], dim=1)[:, None]
        t.apply_edges(fn.u_dot_v("rnorm", "rnorm", "cos_jik"))
        cos_jik = t.edata["cos_jik"]

        # use angle cosine as attention bias? like in ALiBi...
        # or use angle cosine embeddings?
        # with multihead attention, could use relative difference in angle
        # from some set of reference angles...?
        z_jik = self.angle_encoder(cos_jik)

        #
        t.apply_edges(fn.u_add_v("xij", "xij", "x_jik"))
        e_jik = nn.silu(z_jik + t.edata.pop("x_jik"))

        a = (e_jik * self.attn).sum(dim=-1).unsqueeze(dim=2)  # (num_edge, num_heads, 1)

        t.edata["a"] = edge_softmax(t, a)

        g.update_all(fn.u_mul_e("xij", "a", "m"), fn.sum("m", "ft"))

        return g


class TersoffBlock(nn.Module):
    def __init__(self, d_model=256, d_message=64):
        super().__init__()
        self.attention = TersoffAttention()
        self.project_src = nn.Linear(d_model, d_message)
        self.project_dst = nn.Linear(d_model, d_message)
        self.project_edge = nn.Linear(d_model, d_message)

        self.feedforward = FeedForward(d_message, 4 * d_model, d_model)

    def forward(
        self, g: dgl.DGLGraph, t: dgl.DGLGraph, x: torch.Tensor, y: torch.Tensor
    ):
        g = g.local_var()
        t = t.local_var()

        # project down
        g.ndata["xj"] = self.project_src(x)
        g.ndata["xi"] = self.project_dst(x)
        g.apply_edges(fn.u_add_v("xj", "xi", "xij"))
        xij = g.edata.pop("xij") + self.project_edge(y)

        # Dwivedi et al apply the nonlinearity and edge residual
        # before the attention process
        # y = y + F.silu(xij)

        # attention layer
        x = self.attention(g, t, xij)

        # feedforward
        x = self.feedforward(x)

        return x


class TFM(nn.Module):
    """Prototype graph transformer model."""

    def __init__(self, config: TFMConfig = TFMConfig()):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        self.config = config
        self.transform = config.transform

        if config.atom_features == "embedding":
            self.atom_embedding = nn.Embedding(108, config.d_model)
        else:
            f = _get_attribute_lookup(atom_features=config.atom_features)
            self.atom_embedding = nn.Sequential(
                f, MLPLayer(f.embedding_dim, config.d_model)
            )

        self.bond_encoder = RBFExpansion(vmin=0, vmax=8.0, bins=config.d_model)

        self.reference_energy = None
        if config.reference_energies is not None:
            self.reference_energy = nn.Embedding(
                108, embedding_dim=1, _weight=config.reference_energies.view(-1, 1)
            )
            # self.reference_energy.weight.requires_grad_(False)

        self.layers = nn.ModuleList(
            [
                TersoffBlock(config.d_model, config.d_message)
                for idx in range(config.layers)
            ]
        )

        if self.config.energy_units == "eV/atom":
            self.readout = AvgPooling()
        elif self.config.energy_units == "eV":
            self.readout = SumPooling()

        self.fc = nn.Linear(config.d_model, config.output_features)

    @dispatch
    def forward(self, x):
        print("convert")
        return self.forward(Atoms(x))

    @dispatch
    def forward(self, x: Atoms):
        device = next(self.parameters()).device
        return self.forward(self.transform(x).to(device))

    @dispatch
    def forward(
        self,
        g: dgl.DGLGraph,
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata)
        """
        # print("forward")
        config = self.config
        g = g.local_var()

        # to compute forces, take gradient wrt g.edata["r"]
        # need to add bond vectors to autograd graph
        if config.compute_forces:
            g.edata["r"].requires_grad_(True)

        # edge coincidence graph
        t = edge_attention_graph(g)

        # initial node features: atom feature network...
        atomic_number = g.ndata.pop("atomic_number").int()
        x = self.atom_embedding(atomic_number)

        # initial bond features
        bondlength = torch.norm(g.edata["r"], dim=1)
        y = self.bond_encoder(bondlength)
        g.edata["y"] = y

        if config.cutoff is not None:
            # save cutoff function value for application in EdgeGatedGraphconv
            g.edata["cutoff_value"] = self.config.cutoff(bondlength)

        # gated GCN updates: update node, edge features
        for layer in self.layers:
            x = layer(g, t, x, y)

        # predict per-atom energy contribution (in eV)
        atomwise_energy = self.fc(x)
        if self.reference_energy is not None:
            atomwise_energy += self.reference_energy(atomic_number)

        # total energy prediction
        # if config.energy_units = eV/atom, mean reduction
        output = torch.squeeze(self.readout(g, atomwise_energy))

        if config.compute_forces:
            forces, stress = autograd_forces(
                output,
                g.edata["r"],
                g,
                energy_units=config.energy_units,
                compute_stress=True,
            )

            return dict(total_energy=output, forces=forces, stress=stress)

        return output
