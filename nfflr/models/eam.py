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

from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

from nfflr.models.abstract import AbstractModel
from nfflr.atoms import Atoms
from nfflr.graph import periodic_radius_graph
from nfflr.models.utils import autograd_forces


@dataclass
class EAMData:
    cutoff: float
    rs: float
    rhos: float
    embedded_data: float
    density_data: float
    rphi_data: float


def read_setfl(p: Path, comment_rows=3):
    # d = pd.read_csv(p, skiprows=3)
    with open(p, "r") as f:
        for linenum, line in enumerate(f):
            if linenum == comment_rows + 1:
                break

        #  Nrho, drho, Nr, dr, cutoff
        Nrho, drho, Nr, dr, cutoff = line.strip().split()
        Nrho, Nr = int(Nrho), int(Nr)
        drho, dr, cutoff = float(drho), float(dr), float(cutoff)

    rs = torch.tensor(dr * np.arange(Nr), dtype=torch.float)
    rhos = torch.tensor(drho * np.arange(Nrho), dtype=torch.float)

    data = pd.read_csv(p, skiprows=3 + comment_rows, header=None).values.flatten()

    # https://github.com/askhl/ase/blob/master/ase/calculators/eam.py#L311
    # ase "embedded_data"
    embedded = torch.tensor(data[:Nrho], dtype=torch.float)

    # ase density_data
    density = torch.tensor(data[Nrho : Nrho + Nr], dtype=torch.float)

    # ase rphi_data
    # .alloy format only, stored as r * phi
    rphi = torch.tensor(data[Nrho + Nr :], dtype=torch.float)

    return EAMData(cutoff, rs, rhos, embedded, density, rphi)


class TorchEAM(nn.Module):
    """

    currently 4-5x slower than ASE EAM for FCC Al `bulk("Al", "fcc", 4.05)`
    not including the overhead of graph construction
    including that overhead, it's closer to 15x slower...

    should profile where the overhead is

    energy evaluation is close, forces not so close...
    """

    def __init__(self, potential: Path):
        super().__init__()
        self.data = read_setfl(potential)

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
        g.edata["r"].requires_grad_(True)
        r = g.edata["r"]

        bondlen = torch.norm(r, dim=1)

        rho_and_phi = self.radial_spline.evaluate(bondlen)
        rho_and_phi[:, 1] /= bondlen

        g.edata["rho_and_phi"] = rho_and_phi
        g.update_all(fn.copy_e("rho_and_phi", "m"), fn.sum("m", "rho_and_phi"))

        # ensure contiguous inputs to spline evaluation
        F = self.embedding_energy.evaluate(g.ndata["rho_and_phi"][:, 0].contiguous())

        potential_energy = F.sum() + 0.5 * g.ndata["rho_and_phi"][:, 1].sum()

        forces = autograd_forces(potential_energy, r, g, energy_units="eV")

        return potential_energy, forces
