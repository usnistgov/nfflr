"""Neural bond order potential

parameterize a bond order style potential with an ALIGNN network
"""

from typing import Tuple, Union

import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import SumPooling

from pydantic.typing import Literal
from torch import nn
from torch.nn import functional as F

from alignn.models.alignn import MLPLayer, ALIGNNConv, EdgeGatedGraphConv
from alignn.models.utils import RBFExpansion
from alignn.utils import BaseSettings


class BondOrderConfig(BaseSettings):
    """Hyperparameter schema for alignn.models.bond_order."""

    name: Literal["bondorder"]
    alignn_layers: int = 2
    gcn_layers: int = 2
    atom_features: int = 64
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 64
    output_features: Literal[1] = 1
    calculate_gradient: bool = True
    energy_units: Literal["eV", "eV/atom"] = "eV/atom"

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class BondOrderInteraction(nn.Module):
    def __init__(self, node_input_features, cutoff_distance=4, cutoff_onset=3.8):
        super().__init__()
        self.pair_parameters = 4
        self.src_params = nn.Linear(node_input_features, self.pair_parameters)
        self.dst_params = nn.Linear(
            node_input_features, self.pair_parameters, bias=False
        )

        # set bias parameters to log of baseline values?
        # e.g. from Tersoff Si
        # New empirical approach for the structure and energy of covalent systems
        # units are eV and Å⁻¹
        # log([A, λ1, B, λ2])
        self.src_params.bias.data = torch.log(
            # torch.tensor([2280.4, 1.4654, 154.87, 1.4654])  # diamond
            torch.tensor([3264, 3.2394, 95.373, 1.3258])  # Si
        )

        # sinusoidal cutoff function
        D = 0.5 * (cutoff_distance - cutoff_onset)
        R = cutoff_distance - D

        def cutoff(r):
            N = r.size()
            c = torch.where(
                r < R - D,
                torch.ones(N),
                0.5 - 0.5 * torch.sin(np.pi * (r - R) / (2 * D)),
            )
            return torch.where(r > R + D, torch.zeros(N), c)

        self.cutoff = cutoff

    def forward(self, g, node_features, bond_order, bondlength):
        """Bond order style pair interaction

        Inputs:
        g.ndata["x"]: latent atom features
        g.edata["r"]: bond vectors
        g.edata["b"]: predicted bond order
        """

        # TODO: can all this be fused into a single dgl udf? does it matter?
        # predict elementwise interaction parameters from node features
        # all of which need to be positive
        # [A, λ1, B, λ2] <- (exp ∘ Linear)(u || v)
        g.ndata["e_src"] = self.src_params(node_features)
        g.ndata["e_dst"] = self.dst_params(node_features)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "pair_params"))
        params = torch.exp(g.edata.pop("pair_params"))

        # f_repulse(r) =  A exp(-λ1 r)
        f_repulse = params[:, 0] * torch.exp(-params[:, 1] * bondlength)

        # f_attract(r) = B exp(-λ2 r)
        f_attract = params[:, 2] * torch.exp(-params[:, 3] * bondlength)

        V_pair = self.cutoff(bondlength) * (f_repulse - bond_order * f_attract)

        # sum over all bond energies to get per-atom energy contributions
        g.edata["V"] = V_pair
        g.update_all(fn.copy_e("V", "m"), fn.sum("m", "V"))
        return g.ndata.pop("V")


class NeuralBondOrder(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config: BondOrderConfig = BondOrderConfig(name="bondorder")):
        """Initialize class with number of input features, conv layers."""
        super().__init__()

        # just use atom embedding layer
        DICTIONARY_SIZE = 128
        self.atom_embedding = nn.Embedding(DICTIONARY_SIZE, config.atom_features)

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

        self.interaction = BondOrderInteraction(config.atom_features)

        if self.config.energy_units == "eV/atom":
            self.readout = AvgPooling()
        elif self.config.energy_units == "eV":
            self.readout = SumPooling()

        self.fc = nn.Linear(config.hidden_features, config.output_features)

    def forward(self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]):
        """NeuralBondOrder : start with `atom_features`.

        V_ij = f_repulse(r_ij) + b_ij * f_attract(r_ij)

        f_repulse(r) =  A exp(-λ1 r)
        f_attract(r) = -B exp(-λ2 r)

        So the pairwise paramaters [A, B, λ1, λ2] all should be positive

        and the bond order term b_ij is in the range (0, 1), and we can model with
        b_ij = sigmoid ∘ ALIGNN(g, lg)

        r: atomic coordinates (g.ndata)
        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """

        g = g.local_var()

        # initial bond features
        # to compute forces, take gradient wrt g.edata["r"]
        # needs to be included in the graph though...
        r = g.edata["r"]

        if self.config.calculate_gradient:
            r.requires_grad_(True)

        bondlength = torch.norm(r, dim=1)
        g.edata["bondlength"] = bondlength

        # apply GCN to local neighborhoods
        g_local = dgl.edge_subgraph(g, bondlength <= 4.0)

        lg = g_local.line_graph(shared=True)
        lg.apply_edges(compute_bond_cosines)
        z = self.angle_embedding(lg.edata.pop("h"))

        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features").squeeze()
        x = self.atom_embedding(x)
        x_initial = x.clone()

        y = self.edge_embedding(g_local.edata["bondlength"])

        # ALIGNN updates: update node, edge, triplet features
        # print(x.size(), y.size(), z.size())
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g_local, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g_local, x, y)

        # per-bond bond order
        # remove channel dimension...
        bond_order = torch.squeeze(torch.sigmoid(self.fc(y)))

        # potential function reduces edge -> node
        # f(r) = cutoff(r) * (f_repulse(r) + bond_order * f_attract(r))
        # E_i = sum_j(f(r_ij))
        potential = self.interaction(g, x_initial, bond_order, bondlength)

        # use sum pooling to predict total energy
        # for each atomistic configuration in the batch
        energy = self.readout(g, potential)

        forces = torch.empty(1)

        if self.config.calculate_gradient:
            # energy gradient contribution of each bond
            # dU/dr
            dy_dr = grad(
                energy,
                r,
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
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

        return torch.squeeze(energy)
