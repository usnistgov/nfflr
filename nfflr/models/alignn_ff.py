"""Atomistic LIne Graph Neural Network.

A crystal line graph network dgl implementation.
"""
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

from nfflr.graph import compute_bond_cosines
from nfflr.atoms import _get_attribute_lookup


@dataclass
class ALIGNNConfig:
    """Hyperparameter schema for jarvisdgl.models.alignn"""

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

    def forward(
        self,
        g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph],
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        config = self.config

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
        if config.compute_forces:
            r.requires_grad_(True)

        # JVASP-76516_elast-0 is causing issues!
        # i.e. bond length of zero...
        # this is is a precision bug related to distance filtering
        # using float64 and self-interaction threshold 1e-8 works
        # with float32, need to increase the threshold... to 1e-2
        bondlength = torch.norm(r, dim=1)

        if config.cutoff_onset is not None:
            # save cutoff function value for application in EdgeGatedGraphconv
            r_onset, r_cut = config.cutoff_onset, config.cutoff
            fcut = smooth_cutoff(bondlength, r_onset=r_onset, r_cutoff=r_cut)
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

                lg = g_local.line_graph(shared=True)

            # angle features (fixed)
            lg.apply_edges(compute_bond_cosines)
            z = self.angle_embedding(lg.edata.pop("h"))

        # initial node features: atom feature network...
        atomic_number = g.ndata.pop("atomic_number").int().squeeze()
        x = self.atom_embedding(atomic_number)

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
            forces = autograd_forces(output, r, g, energy_units=config.energy_units)

            result = dict(
                total_energy=output,
                forces=forces,
                atomwise_energy=atomwise_energy,
            )
        else:
            result = output

        return result
