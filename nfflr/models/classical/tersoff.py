"""test"""
from dataclasses import dataclass
from typing import Literal, Callable

import numpy as np
import torch
from torch import nn
import dgl
import dgl.function as fn

import nfflr


@dataclass
class TersoffConfig:
    """Tersoff potential parameters.

    values correspond to J. Tersoff, Phys. Rev. B 37, 6991 (1988)
    as documented by https://www.ctcms.nist.gov/potentials/entry/1988--Tersoff-J--Si-b/
    """

    e1: str = "Si"
    e2: str = "Si"
    e3: str = "Si"
    m: Literal[1, 3] = 3
    gamma: float = 1.0
    lambda3: float = 1.3258
    c: float = 4.8381
    d: float = 2.0417
    costheta0: float = 0.0
    n: float = 22.956
    beta: float = 0.33675
    lambda2: float = 1.3258
    B: float = 95.373
    R: float = 3.0
    D: float = 0.2
    lambda1: float = 3.2394
    A: float = 3264.7


def wrap_zeta_message(fcut: Callable, g: Callable, params: TersoffConfig):

    m = params.m
    lambda3 = params.lambda3

    def zeta_message(edges):
        r"""Zeta_ij kernel for Tersoff potential

        ```math
        \sum_{k \neq i, j} fcut(r_{ik}) g(cos(\theta)) exp(\lambda_3^m (r_{ij} - r_{ij})^m)
        ```

        line graph edges: (k -> i) -> (i -> j)

        message propagation updates the (i -> j) bond.
        """

        # relative position vectors for src and dst bonds
        # negate src bond relative position to obtain k <- i -> j
        r_ik = -edges.src["r"]
        r_ij = edges.dst["r"]

        bondlen_ij = torch.norm(r_ij, dim=1)
        bondlen_ik = torch.norm(r_ik, dim=1)

        bond_cosine = torch.sum(r_ik * r_ij, dim=1) / (bondlen_ik * bondlen_ij)
        bond_cosine = torch.clamp(bond_cosine, -1, 1)

        zeta_ij = (
            fcut(bondlen_ik)
            * g(bond_cosine)
            * torch.exp(lambda3**m * (bondlen_ij - bondlen_ik) ** m)
        )

        return {"zeta_ijk": zeta_ij}

    return zeta_message


class Tersoff(nn.Module):
    """Tersoff potential following LAMMPS implementation.

    paper: Tersoff, 1988, 10.1103/PhysRevB.38.9902

    Refer to https://docs.lammps.org/pair_tersoff.html

    """

    def __init__(self, params: TersoffConfig = TersoffConfig()):
        """test"""
        super().__init__()
        self.ps = params
        self.zeta_message = wrap_zeta_message(self.fcut, self.g, self.ps)

    def fcut(self, r: torch.Tensor):
        """Tersoff style smooth cutoff function.

        Parameters
        ----------
        r:
            bond length
        """
        R, D = self.ps.R, self.ps.D

        smoothed = torch.where(
            r < R + D,
            0.5 - 0.5 * torch.sin(np.pi * (r - R) / (2 * D)),
            0,
        )
        return torch.where(r < R - D, 1, smoothed)

    def f_repulsive(self, r):
        """Tersoff pair repulsion term."""
        A, lambda1 = self.ps.A, self.ps.lambda1
        return A * torch.exp(-lambda1 * r)

    def f_attractive(self, r):
        """Tersoff pair attractive term."""
        B, lambda2 = self.ps.B, self.ps.lambda2
        return -B * torch.exp(-lambda2 * r)

    def g(self, angle_cosine):
        """Tersoff angle term."""
        c = self.ps.c
        d = self.ps.d
        gamma = self.ps.gamma
        costheta0 = self.ps.costheta0

        return gamma * (
            1
            + (c**2 / d**2)
            - (c**2 / (d**2 + (angle_cosine - costheta0) ** 2))
        )

    def angular_term(self, g: dgl.DGLGraph):
        r"""Tersoff zeta term, for each edge.

        .. math::

            \sum_{k \neq i, j} \textrm{fcut}(r_{ik}) g(\cos(\theta)) \exp(\lambda_3^m (r_{ij} - r_{ij})^m)

        .. note::

            relies on relative bond vectors :code:`g.edata["r"].requires_grad_ == True`.


        Parameters
        ----------
        g: dgl.DGLGraph
            neighbor list graph


        """
        n = self.ps.n
        beta = self.ps.beta

        # line graph for three-body terms
        # no backtracking edges (k -> i) -> (i -> k)
        # since $k \neq i$ in the sum
        lg = g.line_graph(shared=True, backtracking=False)

        # compute per-bond three-body terms
        # produces zeta_ij edge features
        lg.apply_edges(self.zeta_message)

        # propagate zeta messages to (i -> j) bond features
        lg.update_all(fn.copy_e("zeta_ijk", "m"), fn.sum("m", "zeta_ij"))
        zeta_ij = lg.ndata.pop("zeta_ij")

        b_ij = torch.pow(1 + beta**n * zeta_ij**n, -1 / (2 * n))

        return b_ij

    def forward(self, g: dgl.DGLGraph):
        """Evaluate Tersoff model for DGLGraph `g`."""
        g = g.local_var()
        g.edata["r"].requires_grad_(True)

        # compute bond order terms
        b_ij = self.angular_term(g)

        # compute energy contribution for each bond
        bondlen = torch.norm(g.edata["r"], dim=1)
        energy = (
            self.fcut(bondlen)
            * (self.f_repulsive(bondlen) + b_ij * self.f_attractive(bondlen))
            / 2
        )

        total_energy = energy.sum()

        forces, stress = nfflr.autograd_forces(
            total_energy, g.edata["r"], g, energy_units="eV", compute_stress=True
        )

        return dict(
            total_energy=total_energy,
            forces=forces,
            stress=stress,
        )
