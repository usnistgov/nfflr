"""Atomistic LIne Graph Neural Network.

A crystal line graph network dgl implementation.
"""
from plum import dispatch
from typing import Tuple, Union, Optional, Literal
from dataclasses import dataclass

import dgl
from dgl.nn import AvgPooling, SumPooling

import torch
from torch import nn

from nfflr.models.utils import (
    smooth_cutoff,
    autograd_forces,
    RBFExpansion,
    MLPLayer,
    SparseALIGNNConv,
    EdgeGatedGraphConv,
)

from nfflr.data.graph import compute_bond_cosines, periodic_radius_graph
from nfflr.data.atoms import _get_attribute_lookup, Atoms


@dataclass
class ALIGNNConfig:
    """Hyperparameter schema for nfflr.models.gnn.alignn"""

    cutoff: float = 8.0
    cutoff_onset: Optional[float] = 7.5
    alignn_layers: int = 4
    gcn_layers: int = 4
    atom_features: str = "cgcnn"
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 256
    output_features: int = 1
    compute_forces: bool = False
    energy_units: Literal["eV", "eV/atom"] = "eV/atom"


class ALIGNN(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config: ALIGNNConfig = ALIGNNConfig()):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        self.config = config

        if config.atom_features == "embedding":
            self.atom_embedding = nn.Embedding(108, config.hidden_features)
        else:
            f = _get_attribute_lookup(atom_features=config.atom_features)
            self.atom_embedding = nn.Sequential(
                f, MLPLayer(f.embedding_dim, config.hidden_features)
            )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_input_features),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(vmin=-1, vmax=1.0, bins=config.triplet_input_features),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        width = config.hidden_features
        self.alignn_layers = nn.ModuleList()
        for idx in range(1, config.alignn_layers + 1):
            skipnorm = idx == config.alignn_layers
            self.alignn_layers.append(
                SparseALIGNNConv(width, width, skip_last_norm=skipnorm)
            )

        self.gcn_layers = nn.ModuleList()
        for idx in range(1, config.gcn_layers + 1):
            skipnorm = idx == config.gcn_layers
            self.gcn_layers.append(
                EdgeGatedGraphConv(width, width, skip_edgenorm=skipnorm)
            )

        if self.config.energy_units == "eV/atom":
            self.readout = AvgPooling()
        elif self.config.energy_units == "eV":
            self.readout = SumPooling()

        self.fc = nn.Linear(config.hidden_features, config.output_features)

    @dispatch
    def forward(self, x):
        print("convert")
        return self.forward(Atoms(x))

    @dispatch
    def forward(self, x: Atoms):
        print("construct graph")
        return self.forward(periodic_radius_graph(x, r=self.config.cutoff))

    @dispatch
    def forward(
        self,
        g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph],
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        # print("forward")
        config = self.config

        if isinstance(g, dgl.DGLGraph):
            lg = None
        else:
            g, lg = g

        g = g.local_var()

        # to compute forces, take gradient wrt g.edata["r"]
        # need to add bond vectors to autograd graph
        if config.compute_forces:
            g.edata["r"].requires_grad_(True)

        # initial node features: atom feature network...
        atomic_number = g.ndata.pop("atomic_number").int().squeeze()
        x = self.atom_embedding(atomic_number)

        # initial bond features
        bondlength = torch.norm(g.edata["r"], dim=1)
        y = self.edge_embedding(bondlength)
        g.edata["y"] = y

        if config.cutoff_onset is not None:
            # save cutoff function value for application in EdgeGatedGraphconv
            r_onset, r_cut = config.cutoff_onset, config.cutoff
            fcut = smooth_cutoff(bondlength, r_onset=r_onset, r_cutoff=r_cut)
            g.edata["cutoff_value"] = fcut

        # Local: apply GCN to local neighborhoods
        # solution: index into bond features and turn it into a residual connection?
        local_cutoff = 6.0  # 4.0
        if len(self.alignn_layers) > 0:
            if lg is None:
                g_local = dgl.edge_subgraph(
                    g, bondlength <= local_cutoff, relabel_nodes=False
                )
                if g.num_nodes() != g_local.num_nodes():
                    print("problem with edge_subgraph!")

                lg = g_local.line_graph(shared=True)

            # angle features (fixed)
            lg.apply_edges(compute_bond_cosines)
            z = self.angle_embedding(lg.edata.pop("h"))

        # ALIGNN updates: update node, edge, triplet features
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
        output = torch.squeeze(self.readout(g, atomwise_energy))

        if config.compute_forces:
            forces = autograd_forces(
                output, g.edata["r"], g, energy_units=config.energy_units
            )

            return dict(
                total_energy=output,
                forces=forces,
            )

        return output
