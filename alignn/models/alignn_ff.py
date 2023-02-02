"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""
from typing import Tuple, Union, Optional

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

from alignn.models.utils import (
    smooth_cutoff,
    RBFExpansion,
    MLPLayer,
    SparseALIGNNConv,
    EdgeGatedGraphConv,
)
from alignn.utils import BaseSettings

from jarvis.core.graphs import compute_bond_cosines


class ALIGNNForceFieldConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn_ff"""

    name: Literal["alignn_forcefield"] = "alignn_forcefield"
    cutoff: float = 8.0
    cutoff_onset: Optional[float] = 7.5
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


class ALIGNNForceField(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(
        self,
        config: ALIGNNForceFieldConfig = ALIGNNForceFieldConfig(
            name="alignn_forcefield"
        ),
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
                SparseALIGNNConv(
                    config.hidden_features,
                    config.hidden_features,
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    config.hidden_features, config.hidden_features
                )
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

        # precomputed line graph
        precomputed_lg = False
        if isinstance(g, tuple):
            precomputed_lg = True
            g, lg = g
            lg = lg.local_var()

        g = g.local_var()

        # initial bond features: bond displacement vectors
        r = g.edata["r"]

        # to compute forces, take gradient wrt g.edata["r"]
        # need to add bond vectors to autograd graph
        if self.config.calculate_gradient:
            r.requires_grad_(True)

        # JVASP-76516_elast-0 is causing issues!
        # i.e. bond length of zero...
        # this is is a precision bug related to distance filtering
        # using float64 and self-interaction threshold 1e-8 works
        # with float32, need to increase the threshold... to 1e-2
        bondlength = torch.norm(r, dim=1)

        if self.config.cutoff_onset is not None:
            # save cutoff function value for application in EdgeGatedGraphconv
            fcut = smooth_cutoff(
                bondlength,
                r_onset=self.config.cutoff_onset,
                r_cutoff=self.config.cutoff,
            )
            g.edata["cutoff_value"] = fcut

        # print(bondlength.sort()[0][:10])
        y = self.edge_embedding(bondlength)
        g.edata["y"] = y

        # Local: apply GCN to local neighborhoods
        # problem: need to match number of edges...
        # solution: index into bond features and turn it into a residual connection?
        # crystal graph convolution
        # x, m = self.node_update(g, x, y)
        # line graph convolution:
        # m = m[local_edge_mask]
        # y_update, z = self.edge_update(lg, m, z)
        # y[local_edge_mask] += y_update

        local_cutoff = 4.0
        if len(self.alignn_layers) > 0:
            if not precomputed_lg:
                g_local = dgl.edge_subgraph(
                    g, bondlength <= local_cutoff, relabel_nodes=False
                )
                if g.num_nodes() != g_local.num_nodes():
                    print("problem with edge_subgraph!")
                # y = g_local.edata.pop("y")

                lg = g_local.line_graph(shared=True)

            # angle features (fixed)
            lg.apply_edges(compute_bond_cosines)
            z = self.angle_embedding(lg.edata.pop("h"))

        # initial node features: atom feature network...
        if self.config.sparse_atom_embedding:
            x = g.ndata.pop("atomic_number").long().squeeze()
        else:
            x = g.ndata.pop("atom_features")

        x = self.atom_embedding(x)

        # ALIGNN updates: update node, edge, triplet features
        # print(f"{x.shape=}")
        # print(f"{y.shape=}")
        # print(f"{z.shape=}")
        # print(f"{g=}")
        # print(f"{lg=}")
        y_mask = torch.where(bondlength <= local_cutoff)[0]
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z, y_mask=y_mask)

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
            g.update_all(
                fn.copy_e("pairwise_forces", "m"), fn.sum("m", "forces")
            )

            forces = torch.squeeze(g.ndata["forces"])

            # if training against reduced energies, correct the force predictions
            if self.config.energy_units == "eV/atom":
                # broadcast |v(g)| across forces to under per-atom energy scaling

                n_nodes = torch.cat(
                    [
                        i * torch.ones(i, device=g.device)
                        for i in g.batch_num_nodes()
                    ]
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
                    160.21766208
                    * torch.matmul(r.T, dy_dr)
                    / (2 * g.ndata["V"][0])
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
