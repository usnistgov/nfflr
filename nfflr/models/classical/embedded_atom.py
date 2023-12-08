from plum import dispatch

from pathlib import Path
from typing import Tuple, Union, Optional, Literal
from dataclasses import dataclass

import numpy as np
import pandas as pd

import dgl
import dgl.function as fn
from dgl.nn import AvgPooling, SumPooling

import torch
from torch import nn
from torch.nn.functional import softplus

import orthnet

import nfflr
from nfflr.models.abstract import AbstractModel
from nfflr.data.atoms import Atoms
from nfflr.data.graph import periodic_radius_graph
from nfflr.models.utils import autograd_forces


class ElectronDensity(torch.nn.Module):
    def __init__(self, nbasis: int = 128, cutoff: float = 6.0):
        super().__init__()
        self.nbasis = nbasis
        ps = 0.1 * torch.ones(nbasis)
        self.phi = torch.nn.Parameter(ps)
        # self.rbf = RBFExpansion(vmax=cutoff, bins=nbasis)
        self.activation = torch.nn.functional.softplus # torch.exp

    def reset_parameters(self):
        torch.nn.init.normal_(self.phi.data, 0.0, 0.5)

    # def forward(self, r):
    #     return (0.1 * torch.exp(self.phi) * self.rbf(r)).sum(dim=1)

    def forward(self, r):
        basis = orthnet.Laguerre(r.unsqueeze(1), self.nbasis-1).tensor
        return torch.exp(-r) * (self.activation(self.phi * basis)).sum(dim=1)

class ExponentialRepulsive(torch.nn.Module):
    def __init__(self, amplitude: float = 10.0, lengthscale: float = 1.0):
        super().__init__()

        def inv_softplus(r):
            return r + np.log(-np.expm1(-r))

        self.amplitude = torch.nn.Parameter(torch.tensor(inv_softplus(amplitude)))
        self.lengthscale = torch.nn.Parameter(torch.tensor(inv_softplus(lengthscale)))

    def forward(self, r):
        return softplus(self.amplitude) * torch.exp(-softplus(self.lengthscale) * r)

class EmbeddingFunction(torch.nn.Module):
    def __init__(self, nbasis: int = 8):
        super().__init__()
        self.nbasis = nbasis
        self.weights = torch.nn.Parameter(torch.ones(nbasis))

    def forward(self, r):
        basis = orthnet.Laguerre(r.unsqueeze(1), self.nbasis-1).tensor
        curve = (self.weights * basis).sum(dim=1)
        # subtract sum of parameters to shift the curve so f(0) = 0
        return softplus(curve) - softplus(self.weights.sum())

class EmbeddedAtomPotential(torch.nn.Module):
    def __init__(self, nbasis: int = 128, cutoff: float = 6.0):
        super().__init__()

        self.nbasis = nbasis
        self.cutoff = cutoff

        self.density = ElectronDensity(nbasis=nbasis, cutoff=cutoff)
        self.pair_repulsion = ExponentialRepulsive()
        self.embedding_energy = EmbeddingFunction()

        self.transform = nfflr.nn.PeriodicRadiusGraph(self.cutoff)

    def forward(self, at: nfflr.Atoms):
        g = self.transform(at)

        g = g.local_var()

        # initial bond features: bond displacement vectors
        # need to add bond vectors to autograd graph
        r = g.edata.pop("r")
        r.requires_grad_(True)

        bondlen = torch.norm(r, dim=1)

        g.edata["density_ij"] = self.density(bondlen)
        g.update_all(fn.copy_e("density_ij", "m"), fn.sum("m", "local_density"))
        F = self.embedding_energy(g.ndata["local_density"])
        potential_energy = F.sum() + self.pair_repulsion(bondlen).sum()

        forces = autograd_forces(potential_energy, r, g, energy_units="eV")
        return {"total_energy": potential_energy, "forces": forces}
