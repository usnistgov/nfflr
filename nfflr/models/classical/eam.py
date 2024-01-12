from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

import dgl.function as fn

import torch
from torch import nn

from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

import nfflr


@dataclass
class EAMData:
    cutoff: float
    rs: float
    rhos: float
    embedded_data: float
    density_data: float
    rphi_data: float


def read_setfl(p: Path, comment_rows=3, dtype=torch.float):
    # d = pd.read_csv(p, skiprows=3)
    with open(p, "r") as f:
        for linenum, line in enumerate(f):
            if linenum == comment_rows + 1:
                break

        #  Nrho, drho, Nr, dr, cutoff
        Nrho, drho, Nr, dr, cutoff = line.strip().split()
        Nrho, Nr = int(Nrho), int(Nr)
        drho, dr, cutoff = float(drho), float(dr), float(cutoff)
        # print(f"{Nrho=}, {Nr=}, {drho=}, {dr=}, {cutoff=}")

    rs = torch.tensor(dr * np.arange(Nr), dtype=dtype)
    rhos = torch.tensor(drho * np.arange(Nrho), dtype=dtype)

    data = pd.read_csv(
        p, skiprows=3 + comment_rows, header=None, sep="\s+"
    ).values.flatten()

    # https://github.com/askhl/ase/blob/master/ase/calculators/eam.py#L311
    # ase "embedded_data"
    embedded = torch.tensor(data[:Nrho], dtype=dtype)

    # ase density_data
    density = torch.tensor(data[Nrho : Nrho + Nr], dtype=dtype)

    # ase rphi_data
    # .alloy format only, stored as r * phi
    rphi = torch.tensor(data[Nrho + Nr :], dtype=dtype)

    return EAMData(cutoff, rs, rhos, embedded, density, rphi)


class EAM(nn.Module):
    """Pytorch implementation of (single-element) Embedded Atom Method.

    Currently slower than LAMMPS by about 3.3x (evaluated on 32k atoms)

    LAMMPS does 32k atoms in 29 ms (0.209 s for 7 energy+force evaluations).
    TorchEAM takes 400 ms for neighbors with kd-tree, then 96 ms for energy+forces.

    so LAMMPS is faster by about 3.3x, but LAMMPS neighbor list is *way* faster.
    """

    def __init__(self, potential: Path, dtype=torch.float):
        super().__init__()
        self.data = read_setfl(potential, dtype=dtype)

        # interpolate tabulated data
        d = self.data
        self.embedding_energy = NaturalCubicSpline(
            natural_cubic_spline_coeffs(d.rhos, d.embedded_data.unsqueeze(1))
        )

        # stack electron_density and r * phi splines for more efficient evaluation
        radial_data = torch.hstack(
            (d.density_data.unsqueeze(-1), d.rphi_data.unsqueeze(-1))
        )
        self.radial_spline = NaturalCubicSpline(
            natural_cubic_spline_coeffs(d.rs, radial_data)
        )

    def forward(self, g):  # a: Atoms):
        """EAM

        The energy is decomposed into pairwise ϕ terms and an embedding function U
        of the electron density n around each atom.

        E = \sum_{i < j} Φᵢⱼ(rᵢⱼ) + \sum_i Uᵢ(nᵢ)
        nᵢ = \sum_{j ≠ i} ρⱼ(rᵢⱼ)

        """
        # g = graph.periodic_radius_graph(a, r=self.data.cutoff)
        g = g.local_var()

        # initial bond features: bond displacement vectors
        # need to add bond vectors to autograd graph
        r = g.edata.pop("r")
        r.requires_grad_(True)

        bondlen = torch.norm(r, dim=1)

        # evaluate electron density and pair repulsion at |r|
        # for unary crystals, phi should be symmetric.
        # \rho_ij = spline(r_ij)
        # \phi_ij = spline(r_ij) / |r_ij|
        rho_and_phi = self.radial_spline.evaluate(bondlen)
        rho = rho_and_phi[:, 0]
        phi = rho_and_phi[:, 1] / (2 * bondlen)
        density_ij = rho.detach().contiguous()
        repulsion_ij = phi.detach().contiguous()

        # aggregate local densities and pair interactions over bonds
        g.edata["rho_and_phi"] = torch.hstack((rho.unsqueeze(1), phi.unsqueeze(1)))
        g.update_all(fn.copy_e("rho_and_phi", "m"), fn.sum("m", "rho_and_phi"))
        rho_and_phi = g.ndata.pop("rho_and_phi")

        # ensure contiguous inputs to spline evaluation
        density = rho_and_phi[:, 0].contiguous()
        pair_repulsion = rho_and_phi[:, 1].contiguous()
        F = self.embedding_energy.evaluate(density)

        self.components = {
            "embedding_energy": F.detach(),
            "phi": pair_repulsion.detach(),
            "density": density.detach(),
            "repulsion_ij": repulsion_ij,
            "density_ij": density_ij,
        }
        potential_energy = F.sum() + pair_repulsion.sum()

        forces = nfflr.autograd_forces(potential_energy, r, g, energy_units="eV")
        return potential_energy, forces


# alias
TorchEAM = EAM
