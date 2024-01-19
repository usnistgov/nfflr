"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""

from plum import dispatch

import logging
from dataclasses import dataclass
from typing import Tuple, Union, Optional, Literal, Callable

import dgl
import torch
from torch import nn

from dgl.nn import AvgPooling, SumPooling

import nfflr
from nfflr.nn import (
    RBFExpansion,
    MLPLayer,
    ALIGNNConv,
    EdgeGatedGraphConv,
    AttributeEmbedding,
    PeriodicRadiusGraph,
    XPLOR,
)
from nfflr.data.graph import compute_bond_cosines


@dataclass
class ALIGNNConfig:
    """Hyperparameter schema for nfflr.models.gnn.alignn."""

    transform: Callable = PeriodicRadiusGraph(cutoff=8.0)
    # cutoff: Optional[tuple[float]] = (7.5, 8.0)
    cutoff: torch.nn.Module = XPLOR(7.5, 8.0)
    alignn_layers: int = 4
    gcn_layers: int = 4
    norm: Literal["batchnorm", "layernorm"] = "batchnorm"
    atom_features: str = "cgcnn"
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 256
    output_features: int = 1
    compute_forces: bool = False
    energy_units: Literal["eV", "eV/atom"] = "eV/atom"
    reference_energies: Optional[torch.Tensor] = None


class ALIGNN(torch.nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config: ALIGNNConfig = ALIGNNConfig()):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        self.config = config
        self.transform = config.transform
        logging.debug(f"{config=}")

        if config.atom_features == "embedding":
            self.atom_embedding = torch.nn.Embedding(108, config.hidden_features)
        else:
            self.atom_embedding = AttributeEmbedding(
                config.atom_features, d_model=config.hidden_features
            )

        self.reference_energy = None
        if config.reference_energies is not None:
            self.reference_energy = torch.nn.Embedding(
                108, embedding_dim=1, _weight=config.reference_energies.view(-1, 1)
            )

        self.edge_embedding = torch.nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
            ),
            MLPLayer(
                config.edge_input_features, config.embedding_features, norm=config.norm
            ),
            MLPLayer(
                config.embedding_features, config.hidden_features, norm=config.norm
            ),
        )
        self.angle_embedding = torch.nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            MLPLayer(
                config.triplet_input_features,
                config.embedding_features,
                norm=config.norm,
            ),
            MLPLayer(
                config.embedding_features, config.hidden_features, norm=config.norm
            ),
        )

        self.alignn_layers = torch.nn.ModuleList(
            [
                ALIGNNConv(
                    config.hidden_features, config.hidden_features, norm=config.norm
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = torch.nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    config.hidden_features, config.hidden_features, norm=config.norm
                )
                for idx in range(config.gcn_layers)
            ]
        )

        if config.energy_units == "eV/atom":
            self.readout = AvgPooling()
        else:
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
    def forward(self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
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
        atomic_number = g.ndata.pop("atomic_number").int()
        x = self.atom_embedding(atomic_number)

        # initial bond features
        bondlength = torch.norm(g.edata["r"], dim=1)
        y = self.edge_embedding(bondlength)

        if config.cutoff is not None:
            # save cutoff function value for application in EdgeGatedGraphconv
            g.edata["cutoff_value"] = self.config.cutoff(bondlength)

        # initial triplet features
        if len(self.alignn_layers) > 0:
            if lg is None:
                lg = g.line_graph(shared=True)

            lg.apply_edges(compute_bond_cosines)
            z = self.angle_embedding(lg.edata.pop("h"))

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # norm-activation-pool-classify
        h = self.readout(g, x)
        output = torch.squeeze(self.fc(h))

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
