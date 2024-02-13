import logging
from dataclasses import dataclass
from typing import Optional, Literal, Callable

import torch
import numpy as np

import dgl
import dgl.function as fn
from dgl.nn import AvgPooling, SumPooling

import nfflr
from nfflr.nn import (
    Norm,
    FeedForward,
    RBFExpansion,
    AttributeEmbedding,
    PeriodicTableEmbedding,
    PeriodicRadiusGraph,
    XPLOR,
)


class DepthwiseConv(torch.nn.Module):
    def __init__(self, d_in, d_radial):
        super().__init__()
        self.radial_filters = torch.nn.Linear(d_radial, d_in)
        self.reset_parameters()

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor, edge_basis: torch.Tensor):
        with g.local_scope():
            g.srcdata["hv"] = x
            g.edata["filter"] = self.radial_filters(edge_basis)
            g.update_all(fn.u_mul_e("hv", "filter", "m"), fn.sum("m", "h"))
            return g.dstdata["h"]

    def reset_parameters(self):
        # initialize filter-generating layer parameters
        # so feature variance at initialization after aggregation is roughly 1.
        # this may be dataset/cutoff dependent...
        torch.nn.init.kaiming_normal_(
            self.radial_filters.weight, nonlinearity="linear", mode="fan_in"
        )
        torch.nn.init.zeros_(self.radial_filters.bias)


class CFConv(torch.nn.Module):
    def __init__(self, d_in, d_radial, d_hidden, d_out=None):
        """CFConv"""
        super().__init__()

        if d_out is None:
            d_out = d_in

        self.pre = torch.nn.Linear(d_in, d_hidden)
        self.depthwise = DepthwiseConv(d_hidden, d_radial)
        self.post = torch.nn.Sequential(
            torch.nn.Linear(d_hidden, d_out),
            torch.nn.SiLU(),
        )
        self.reset_parameters()

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor, edge_basis: torch.Tensor):
        x = self.pre(x)
        x = self.depthwise(g, x, edge_basis)
        return self.post(x)

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.pre.weight, nonlinearity="linear")
        torch.nn.init.zeros_(self.pre.bias)
        torch.nn.init.kaiming_normal_(self.post[0].weight, nonlinearity="relu")
        torch.nn.init.zeros_(self.post[0].bias)


class CFConvFused(torch.nn.Module):
    def __init__(self, d_in, d_radial, d_hidden, d_out=None):
        """CFConv."""
        super().__init__()

        if d_out is None:
            d_out = d_in

        self.pre = torch.nn.Linear(d_in, d_hidden)
        self.radial_filters = torch.nn.Linear(d_radial, d_hidden)
        self.post = torch.nn.Sequential(
            torch.nn.Linear(d_hidden, d_out), torch.nn.SiLU()
        )

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor, edge_basis: torch.Tensor):
        with g.local_scope():
            g.srcdata["hv"] = self.pre(x)
            g.edata["filter"] = self.radial_filters(edge_basis)
            g.update_all(fn.u_mul_e("hv", "filter", "m"), fn.sum("m", "h"))
            return self.post(g.dstdata["h"])


class CFBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_radial: int,
        d_hidden: Optional[int] = None,
        norm: Literal["layernorm", "batchnorm"] = "layernorm",
    ):
        """pre-normalization CFConv + FeedForward block."""
        super().__init__()
        self.prenorm = Norm(d_model, norm)
        self.conv = CFConv(d_model, d_radial, d_hidden)
        self.feedforward = FeedForward(d_model, norm=False)

    def forward(self, g, x, radial_basis):
        identity = x
        x = self.prenorm(x)
        x = self.conv(g, x, radial_basis) + identity
        identity = x
        return self.feedforward(x) + identity


@dataclass
class SchNetConfig:
    """Hyperparameter schema for nfflr.models.gnn.alignn."""

    transform: Callable = PeriodicRadiusGraph(cutoff=5.0)
    cutoff: torch.nn.Module = XPLOR(7.5, 8.0)
    layers: int = 4
    norm: Optional[Literal["batchnorm", "layernorm"]] = None
    atom_features: str | torch.nn.Module = "cgcnn"
    edge_input_features: int = 128
    d_model: int = 128
    output_features: int = 1
    compute_forces: bool = False
    energy_units: Literal["eV", "eV/atom"] = "eV/atom"
    reference_energies: Optional[torch.Tensor] = None
    initialize_bias: bool = False


class SchNet(torch.nn.Module):
    def __init__(self, config: SchNetConfig = SchNetConfig()):
        super().__init__()
        self.config = config
        self.transform = config.transform
        logging.debug(f"{config=}")

        if config.atom_features == "embedding":
            self.atom_embedding = PeriodicTableEmbedding(config.d_model)
        else:
            self.atom_embedding = AttributeEmbedding(
                config.atom_features, config.d_model
            )

        self.reference_energy = None
        if config.reference_energies is not None:
            self.reference_energy = torch.nn.Embedding(
                108, embedding_dim=1, _weight=config.reference_energies.view(-1, 1)
            )
            # self.reference_energy.weight.requires_grad_(False)

        self.edge_basis = RBFExpansion(
            vmin=0,
            vmax=self.config.cutoff.r_cutoff,
            bins=config.edge_input_features,
        )

        self.blocks = torch.nn.ModuleList(
            [
                CFBlock(
                    config.d_model,
                    config.edge_input_features,
                    config.d_model,
                    norm=config.norm,
                )
                for idx in range(config.layers)
            ]
        )

        self.postnorm = Norm(config.d_model, config.norm)
        # self.postnorm = torch.nn.LayerNorm(config.d_model, elementwise_affine=False)

        if config.energy_units == "eV/atom":
            self.readout = AvgPooling()
        else:
            self.readout = SumPooling()

        self.fc = torch.nn.Linear(config.d_model, config.output_features)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(
            self.fc.weight, nonlinearity="linear", mode="fan_in"
        )
        torch.nn.init.zeros_(self.fc.bias)

    def reset_output_scale(self, avg_num_nodes: int):
        """Initialize output layer so average total energy at init has std 1."""
        var = 1 / (avg_num_nodes * self.config.d_model)
        torch.nn.init.normal_(self.fc.weight, std=np.sqrt(var))
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, g: nfflr.Atoms | dgl.DGLGraph):
        config = self.config
        if isinstance(g, nfflr.Atoms):
            g = self.transform(g)

        # to compute forces, take gradient wrt g.edata["r"]
        # need to add bond vectors to autograd graph
        if config.compute_forces:
            g.edata["r"].requires_grad_(True)

        x = self.atom_embedding(g.ndata["atomic_number"])
        radial_basis = self.edge_basis(torch.norm(g.edata["r"], dim=1))

        identity = x
        for cfblock in self.blocks:
            x = cfblock(g, x, radial_basis) + identity

        # norm-linear-sum
        if config.norm:
            x = self.postnorm(x)
        x = self.fc(x)

        if self.reference_energy is not None:
            x += self.reference_energy(g.ndata["atomic_number"])

        output = torch.squeeze(self.readout(g, x))

        if config.compute_forces:
            forces, stress = nfflr.autograd_forces(
                output,
                g.edata["r"],
                g,
                energy_units=config.energy_units,
                compute_stress=True,
            )

            return dict(
                total_energy=output, forces=forces, stress=stress, x=x.squeeze()
            )

        return output
