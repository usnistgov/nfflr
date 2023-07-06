from pathlib import Path
from typing import Tuple, Union, Optional, Literal
from dataclasses import dataclass

import torch
from torch import nn
from torch.autograd import grad

import numpy as np

import dgl
import dgl.function as fn
from dgl.nn import AvgPooling, SumPooling

import pykeops
from pykeops.torch import LazyTensor

# from nfflr.models.abstract import AbstractModel
from nfflr.data.atoms import Atoms
from nfflr.data.graph import periodic_radius_graph, pad_ghost_region
from nfflr.models.utils import autograd_forces


def cutoff_function(r: float, rc: float, ro: float):
    """Smooth cutoff function from ASE Lennard Jones

    Goes from 1 to 0 between ro and rc, ensuring
    that u(r) = lj(r) * cutoff_function(r) is C^1.

    Defined as 1 below ro, 0 above rc.

    Note that r, rc, ro are all expected to be squared,
    i.e. `r = r_ij^2`, etc.

    follows ASE Lennard jones, which in turn follows https://github.com/google/jax-md.

    """

    return torch.where(
        r < ro,
        1.0,
        torch.where(
            r < rc, (rc - r) ** 2 * (rc + 2 * r - 3 * ro) / (rc - ro) ** 3, 0.0
        ),
    )


@dataclass
class LJParams:
    epsilon: float = 1.0
    sigma: float = 1.0
    rc: float = 3.0
    ro: float = 1.98
    smooth: bool = False


class LennardJonesK(nn.Module):
    """KeOps implementation of lennard jones potential."""

    def __init__(self, parameters: LJParams = LJParams()):
        super().__init__()
        self.parameters = parameters

    def forward(self, a: Atoms):
        ps = self.parameters
        if ps.smooth:
            raise NotImplementedError("cutoff function not implemented")

        offsets, atom_ids = pad_ghost_region(a)
        root_cell = (offsets == 0).all(dim=1).nonzero().flatten()

        _xs = a.positions @ a.lattice
        _xs.requires_grad_(True)

        xs = offsets @ a.lattice + _xs[atom_ids]

        N, D = xs.shape

        x_i = LazyTensor(xs.view(N, 1, D))  # (N, 1, D)
        x_j = LazyTensor(xs.view(1, N, D))  # (1, N, D)

        # pair distance - for window and also for basis functions
        d2_ij = ((x_i - x_j) ** 2).sum(-1)  # (N, N, 1)

        # d2_ij is zero along the diagonal
        # negate to get the diagonal in the first branch (x >= 0)
        # positive distances in the second branch
        # c6 = (ps.sigma ** 2 / d2_ij) ** 3    # (N, N, 1)
        c6 = (-d2_ij).ifelse(0, ((ps.sigma**2) * d2_ij) ** -3)

        c12 = c6**2  # (N, N, 1)

        pairwise_energies = 4 * ps.epsilon * (c12 - c6)  # (N, N, 1)

        energy_i = (0.5 * pairwise_energies).sum(dim=1)  # (N, 1, 1)

        # slice only (000) cell
        energy_i = energy_i[root_cell]

        total_energy = energy_i.sum()

        # force calculation based on position
        # retain graph for displacement-based grad calculation
        forces_x = -torch.autograd.grad(
            total_energy, _xs, create_graph=True, retain_graph=True
        )[0]

        return total_energy, forces_x


class LennardJones(nn.Module):
    """DGL implementation of lennard jones potential."""

    def __init__(self, parameters: LJParams = LJParams()):
        super().__init__()
        self.parameters = parameters

    def forward(self, a: Atoms):
        """LennardJones

        The energy is decomposed into pairwise ϕ terms

        E = \sum_{i < j} Φᵢⱼ(rᵢⱼ)
        """
        ps = self.parameters
        if ps.smooth:
            raise NotImplementedError("cutoff function not implemented")

        g = periodic_radius_graph(a, r=self.parameters.rc)
        g = g.local_var()

        # potential value at rc
        e0 = 4 * ps.epsilon * ((ps.sigma / ps.rc) ** 12 - (ps.sigma / ps.rc) ** 6)

        # initial bond features: bond displacement vectors
        # need to add bond vectors to autograd graph
        bond_vectors = g.edata.pop("r")
        bond_vectors.requires_grad_(True)
        # bondlen = torch.norm(bond_vectors, dim=1)
        r2 = (bond_vectors**2).sum(dim=1)

        c6 = (ps.sigma**2 / r2) ** 3
        c12 = c6**2
        pairwise_energies = 4 * ps.epsilon * (c12 - c6)
        analytic_pairwise_forces = -24 * ps.epsilon * (2 * c12 - c6) / r2  # du_ij

        # cutoff correction...
        pairwise_energies -= e0 * (c6 != 0.0)

        # reduce pair potential from edges -> atoms
        # (src -> dst) :: energy[src] ++
        # each edge counts for half the energy since LJ is defined
        # in a way that does not double-count bonds...
        g.edata["pair_energy"] = 0.5 * pairwise_energies
        g.update_all(fn.copy_e("pair_energy", "m"), fn.sum("m", "energy"))

        potential_energy = g.ndata["energy"].sum()
        forces = autograd_forces(potential_energy, bond_vectors, g, energy_units="eV")

        g.edata["analytic_pair_forces"] = (
            analytic_pairwise_forces.unsqueeze(-1) * bond_vectors
        )
        g.update_all(
            fn.copy_e("analytic_pair_forces", "m"), fn.sum("m", "analytic_force")
        )

        return potential_energy, forces, g.ndata["analytic_force"]
