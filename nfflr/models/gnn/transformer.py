from plum import dispatch
from typing import Optional, Literal, Callable
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch import nn

import dgl
import dgl.function as fn
from dgl.nn import AvgPooling, SumPooling
from dgl.nn.functional import edge_softmax

import nfflr
from nfflr.nn import (
    RBFExpansion,
    ChebyshevExpansion,
    MLPLayer,
    AttributeEmbedding,
    PeriodicRadiusGraph,
    XPLOR,
)


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

    transform: Callable = PeriodicRadiusGraph(cutoff=5.0)
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
        e_jik = F.silu(z_jik + t.edata.pop("x_jik"))

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
            self.atom_embedding = torch.nn.Embedding(108, config.d_model)
        else:
            self.atom_embedding = AttributeEmbedding(
                config.atom_features, config.d_model
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
        return self.forward(nfflr.Atoms(x))

    @dispatch
    def forward(self, x: nfflr.Atoms):
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
        t = edge_attention_graph(g, shared=True)

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
            forces, stress = nfflr.autograd_forces(
                output,
                g.edata["r"],
                g,
                energy_units=config.energy_units,
                compute_stress=True,
            )

            return dict(total_energy=output, forces=forces, stress=stress)

        return output
