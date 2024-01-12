from plum import dispatch
from typing import Optional, Literal, Callable
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch import nn

import dgl
import dgl.function as fn
from dgl.nn import AvgPooling, SumPooling

import nfflr
from nfflr.nn import (
    RBFExpansion,
    MLPLayer,
    Norm,
    EdgeGatedGraphConv,
    AttributeEmbedding,
    PeriodicRadiusGraph,
    XPLOR,
)


@dataclass
class TFMConfig:
    """Hyperparameter schema for nfflr.models.gnn.tfm"""

    transform: Callable = PeriodicRadiusGraph(cutoff=8.0)
    cutoff: torch.nn.Module = XPLOR(7.5, 8.0)
    layers: int = 3
    norm: Literal["layernorm", "instancenorm"] = "layernorm"
    atom_features: str = "embedding"
    edge_input_features: int = 80
    embedding_features: int = 64
    hidden_features: int = 256
    output_features: int = 1
    compute_forces: bool = False
    energy_units: Literal["eV", "eV/atom"] = "eV/atom"
    reference_energies: Optional[torch.Tensor] = None


class FeedForward(nn.Module):
    """feedforward layer."""

    def __init__(self, in_features: int, hidden_ratio: int = 4):
        """Linear SiLU Linear feedforward layer."""
        super().__init__()
        hidden_features = hidden_ratio * in_features
        self.layer = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, in_features),
        )

    def forward(self, x):
        """Linear, norm, silu layer."""
        return self.layer(x)


# class EdgeGatedGraphConv(nn.Module):
#     """Edge gated graph convolution from arxiv:1711.07553.

#     see also https://www.jmlr.org/papers/v24/22-0567.html


#     This is similar to CGCNN, but edge features only go into
#     the soft attention / edge gating function, and the primary
#     node update function is W cat(u, v) + b
#     """

#     def __init__(
#         self,
#         input_features: int,
#         output_features: int,
#         norm: Literal["layernorm", "instancenorm", "batchnorm"] = "layernorm",
#     ):
#         """Edge gated graph convolution variant."""
#         super().__init__()

#         # CGCNN-Conv operates on augmented edge features
#         # z_ij = cat(v_i, v_j, u_ij)
#         # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
#         # coalesce parameters for W_f and W_s
#         # but -- split them up along feature dimension
#         self.src_gate = nn.Linear(input_features, output_features)
#         self.dst_gate = nn.Linear(input_features, output_features)
#         self.edge_gate = nn.Linear(input_features, output_features)
#         self.norm_edges = Norm(output_features, norm_type=norm, mode="edge")

#         self.src_update = nn.Linear(input_features, output_features)
#         self.dst_update = nn.Linear(input_features, output_features)
#         self.norm_nodes = Norm(output_features, norm_type=norm, mode="node")

#     def forward(
#         self,
#         g: dgl.DGLGraph,
#         node_feats: torch.Tensor,
#         edge_feats: torch.Tensor,
#     ) -> torch.Tensor:
#         """Edge-gated graph convolution.

#         h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
#         """
#         g = g.local_var()

#         # instead of concatenating (u || v || e) and applying one weight matrix
#         # split the weight matrix into three, apply, then sum
#         # see https://docs.dgl.ai/guide/message-efficient.html
#         # but split them on feature dimensions to update u, v, e separately
#         # m = BatchNorm(Linear(cat(u, v, e)))

#         # compute edge updates, equivalent to:
#         # Softplus(Linear(u || v || e))
#         g.ndata["e_src"] = self.src_gate(node_feats)
#         g.ndata["e_dst"] = self.dst_gate(node_feats)
#         g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))

#         m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)
#         # Dwivedi et al apply the nonlinearity and edge residual
#         # before the attention process
#         y = edge_feats + F.silu(self.norm_edges(g, m))

#         # if edge attributes have a cutoff function value
#         # multiply the edge gate values with the cutoff value
#         cutoff_value = g.edata.get("cutoff_value")
#         if cutoff_value is not None:
#             g.edata["sigma"] = torch.sigmoid(y) * cutoff_value.unsqueeze(1)
#         else:
#             g.edata["sigma"] = torch.sigmoid(y)

#         # compute pair interaction modulated by edge gates
#         g.ndata["Bh"] = self.dst_update(node_feats)
#         g.update_all(fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h"))
#         g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
#         g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
#         x = self.src_update(node_feats) + g.ndata.pop("h")

#         # node and edge updates
#         x = node_feats + F.silu(self.norm_nodes(g, x))

#         return x, y


class TFM(nn.Module):
    """Prototype graph transformer model."""

    def __init__(self, config: TFMConfig = TFMConfig()):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        self.config = config
        self.transform = config.transform

        if config.atom_features == "embedding":
            self.atom_embedding = nn.Embedding(108, config.hidden_features)
        else:
            self.atom_embedding = AttributeEmbedding(
                config.atom_features, config.hidden_features
            )

        self.edge_embedding = RBFExpansion(
            vmin=0, vmax=8.0, bins=config.hidden_features
        )

        self.reference_energy = None
        if config.reference_energies is not None:
            self.reference_energy = nn.Embedding(
                108, embedding_dim=1, _weight=config.reference_energies.view(-1, 1)
            )
            # self.reference_energy.weight.requires_grad_(False)

        width = config.hidden_features

        self.gcn_layers = nn.ModuleList(
            [EdgeGatedGraphConv(width, width) for idx in range(config.layers)]
        )

        if self.config.energy_units == "eV/atom":
            self.readout = AvgPooling()
        elif self.config.energy_units == "eV":
            self.readout = SumPooling()

        self.fc = nn.Linear(config.hidden_features, config.output_features)

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

        # initial node features: atom feature network...
        atomic_number = g.ndata.pop("atomic_number").int()
        x = self.atom_embedding(atomic_number)

        # initial bond features
        bondlength = torch.norm(g.edata["r"], dim=1)
        y = self.edge_embedding(bondlength)
        g.edata["y"] = y

        if config.cutoff is not None:
            # save cutoff function value for application in EdgeGatedGraphconv
            g.edata["cutoff_value"] = self.config.cutoff(bondlength)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

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
