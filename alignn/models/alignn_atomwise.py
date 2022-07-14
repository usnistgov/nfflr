"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""
from typing import Tuple, Union

import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling, SumPooling

# from dgl.nn.functional import edge_softmax
from pydantic.typing import Literal
from torch import nn
from torch.autograd import grad
from torch.nn import functional as F

from alignn.models.utils import RBFExpansion
from alignn.utils import BaseSettings

from jarvis.core.graphs import compute_bond_cosines


class ALIGNNAtomWiseConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["alignn_atomwise"] = "alignn_atomwise"
    alignn_layers: int = 4
    gcn_layers: int = 4
    atom_input_features: int = 92
    sparse_atom_embedding: bool = False
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 256
    output_features: int = 1
    calculate_gradient: bool = True
    calculate_stress: bool = False
    energy_units: Literal["eV", "eV/atom"] = "eV/atom"

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.LayerNorm(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.LayerNorm(output_features)

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

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h"))
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # softmax version seems to perform slightly worse
        # that the sigmoid-gated version
        # compute node updates
        # Linear(u) + edge_gates ⊙ Linear(v)
        # g.edata["gate"] = edge_softmax(g, y)
        # g.ndata["h_dst"] = self.dst_update(node_feats)
        # g.update_all(fn.u_mul_e("h_dst", "gate", "m"), fn.sum("m", "h"))
        # x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z)

        return x, y, z


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)


class ALIGNNAtomWise(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(
        self,
        config: ALIGNNAtomWiseConfig = ALIGNNAtomWiseConfig(name="alignn_atomwise"),
    ):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        self.config = config

        if config.sparse_atom_embedding:
            self.atom_embedding = nn.Sequential(
                nn.Embedding(108, config.embedding_features),
                MLPLayer(config.embedding_features, config.hidden_features),
            )
        else:
            self.atom_embedding = MLPLayer(
                config.atom_input_features, config.hidden_features
            )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
            ),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    config.hidden_features,
                    config.hidden_features,
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(config.hidden_features, config.hidden_features)
                for idx in range(config.gcn_layers)
            ]
        )

        if self.config.energy_units == "eV/atom":
            self.readout = AvgPooling()
        elif self.config.energy_units == "eV":
            self.readout = SumPooling()

        self.fc = nn.Linear(config.hidden_features, config.output_features)

    def forward(
        self,
        g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph],
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """

        if len(self.alignn_layers) > 0:
            g, lg = g
            lg = lg.local_var()
            lg.apply_edges(compute_bond_cosines)

            # angle features (fixed)
            z = self.angle_embedding(lg.edata.pop("h"))

        g = g.local_var()

        # initial node features: atom feature network...
        if self.config.sparse_atom_embedding:
            x = g.ndata.pop("atomic_number").long().squeeze()
        else:
            x = g.ndata.pop("atom_features")

        x = self.atom_embedding(x)
        r = g.edata["r"]

        if self.config.calculate_gradient:
            r.requires_grad_(True)

        # JVASP-76516_elast-0 is causing issues!
        # i.e. bond length of zero...
        # this is is a precision bug related to distance filtering
        # using float64 and self-interaction threshold 1e-8 works
        # with float32, need to increase the threshold... to 1e-2
        bondlength = torch.norm(r, dim=1)
        # print(bondlength.sort()[0][:10])
        y = self.edge_embedding(bondlength)

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # predict per-atom energy contribution (in eV)
        atomwise_energy = self.fc(x)

        # total energy prediction
        # if config.energy_units = eV/atom, mean reduction
        total_energy = torch.squeeze(self.readout(g, atomwise_energy))

        forces = torch.empty(1)
        stress = torch.empty(1)

        if self.config.calculate_gradient:
            # potentially we only need to build the computational graph
            # for the forces at training time, so that we can compute
            # the gradient of the force (and stress) loss?
            create_graph = True

            # energy gradient contribution of each bond
            # dU/dr
            dy_dr = grad(
                total_energy,
                r,
                grad_outputs=torch.ones_like(total_energy),
                create_graph=create_graph,
                retain_graph=True,
            )[0]

            # forces: negative energy gradient -dU/dr
            pairwise_forces = -dy_dr

            # reduce over bonds to get forces on each atom
            g.edata["pairwise_forces"] = pairwise_forces
            g.update_all(fn.copy_e("pairwise_forces", "m"), fn.sum("m", "forces"))

            forces = torch.squeeze(g.ndata["forces"])

            # if training against reduced energies, correct the force predictions
            if self.config.energy_units == "eV/atom":
                # broadcast |v(g)| across forces to under per-atom energy scaling

                n_nodes = torch.cat(
                    [i * torch.ones(i, device=g.device) for i in g.batch_num_nodes()]
                )

                forces = forces * n_nodes[:, None]

            if self.config.calculate_stress:
                # make this a custom DGL aggregation?

                # Under development, use with caution
                # 1 eV/Angstrom3 = 160.21766208 GPa
                # 1 GPa = 10 kbar
                # Following Virial stress formula, assuming inital velocity = 0
                # Save volume as g.gdta['V']?
                stress = -1 * (
                    160.21766208 * torch.matmul(r.T, dy_dr) / (2 * g.ndata["V"][0])
                )
                # virial = (
                #    160.21766208
                #    * 10
                #    * torch.einsum("ij, ik->jk", result["r"], result["dy_dr"])
                #    / 2
                # )  # / ( g.ndata["V"][0])

        result = dict(
            total_energy=total_energy,
            forces=forces,
            stress=stress,
            atomwise_energy=atomwise_energy,
        )

        return result
