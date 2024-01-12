from typing import Literal

import numpy as np

import dgl
import dgl.function as fn

import torch

# from torch import nn
from torch.nn.functional import softplus

import orthnet

import nfflr
from nfflr.nn import RBFExpansion


class GaussianSpline(torch.nn.Module):
    def __init__(
        self,
        nbasis: int = 128,
        cutoff: float = 6.0,
        activation: Literal["softplus"] | None = None,
    ):
        super().__init__()
        self.nbasis = nbasis
        self.cutoff = cutoff
        self.activation = torch.nn.Identity()

        ps = 0.1 * torch.ones(nbasis)
        self.phi = torch.nn.Parameter(ps)
        self.basis = RBFExpansion(vmax=cutoff, bins=nbasis)

        if activation == "softplus":
            self.activation = torch.nn.Softplus()
        self.reset_parameters()

    def fcut(self, r):
        return (1 + torch.cos(np.pi * r / self.cutoff)) / 2

    def scale_distance(self, r, ls=5.0):
        """Distance scaling function from ACE (10.1103/PhysRevB.99.014104)."""
        return 1 - 2 * (
            torch.divide(torch.exp(-ls * ((r / self.cutoff) - 1)) - 1, np.exp(ls) - 1)
        )

    def reset_parameters(self):
        torch.nn.init.normal_(self.phi.data, 0.0, 0.5)

    def forward(self, r):
        b = self.basis(r) * self.fcut(r).unsqueeze(1) / 2
        # b = (1 - self.basis(self.scale_distance(r))) * self.fcut(r).unsqueeze(1) / 2
        return (self.activation(self.phi) * b).sum(dim=1)


class ExponentialRepulsive(torch.nn.Module):
    def __init__(self, amplitude: float = 2e5, lengthscale: float = 5.0):
        super().__init__()

        def inv_softplus(r):
            return r + np.log(-np.expm1(-r))

        self.amplitude = torch.nn.Parameter(torch.tensor(inv_softplus(amplitude)))
        self.lengthscale = torch.nn.Parameter(torch.tensor(inv_softplus(lengthscale)))

    def forward(self, r):
        return softplus(self.amplitude) * torch.exp(-softplus(self.lengthscale) * r)


class LaguerreRepulsive(torch.nn.Module):
    def __init__(self, nbasis: int = 8, cutoff: float = 6):
        super().__init__()
        self.nbasis = nbasis
        self.lengthscale = 2.0

        ps = torch.zeros(nbasis)
        self.phi = torch.nn.Parameter(ps)
        self.activation = torch.nn.functional.softplus  # torch.exp

    def forward(self, r):

        basis = orthnet.Laguerre(r.unsqueeze(1), self.nbasis - 1).tensor
        return torch.exp(-r / 2) * (self.activation(self.phi * basis)).sum(dim=1)

        # return softplus(self.amplitude) * torch.exp(self.lengthscale * r)


class PolynomialEmbeddingFunction(torch.nn.Module):
    def __init__(self, degree: int = 4, use_sqrt_term: bool = True):
        super().__init__()
        self.degree = degree
        self.use_sqrt_term = use_sqrt_term

        # polynomial expansion terms - omit constant offset to enforce F(0) = 0
        powers = 1.0 + torch.arange(degree)

        # scale coefficients for numerical reasons
        scalefactors = 1 / 10 ** torch.cumsum(torch.log10(powers), dim=0)

        if use_sqrt_term:
            # initialize to F(rho) = sqrt(rho)
            powers = torch.hstack((0.5 * torch.ones(1), powers))
            scalefactors = torch.hstack((2 * torch.ones(1), scalefactors))
            init_weights = torch.hstack((-torch.ones(1) / 2, torch.zeros(degree)))
        else:
            # initialize to near-linear function with small curvature...
            init_weights = torch.hstack(
                (torch.tensor([-1, 0.1]), torch.zeros(degree - 2))
            )

        self.powers = powers
        self.scalefactors = scalefactors
        self.weights = torch.nn.Parameter(init_weights)

    def forward(self, density):
        """Evaluate polynomial."""
        return density.unsqueeze(-1).pow(self.powers) @ (
            self.weights * self.scalefactors
        )

    def curvature(self, density):
        """Analytic curvature of embedding function for regularization."""

        # multiplicative factors from polynomial second derivatives...
        mult = (self.powers * (1 + self.powers))[:-1]

        if self.use_sqrt_term:
            mask = torch.ones(self.degree + 1, dtype=bool)
            mask[1] = 0

            # overwrite the factor for the sqrt term
            mult[0] = -1 / 4

            w = self.weights[mask]
            sf = mult * self.scalefactors[mask]
            p = self.powers[mask] - 2

        else:
            w = self.weights[1:]
            sf = mult * self.scalefactors[1:]
            p = self.powers[1:] - 2

        curve = density.unsqueeze(-1).pow(p) @ (w * sf)
        return curve


class SplineEmbeddingFunction(torch.nn.Module):
    def __init__(self, nbasis: int = 8):
        super().__init__()
        self.nbasis = nbasis
        self.weights = torch.nn.Parameter(-0.1 * torch.ones(nbasis))
        self.basis = RBFExpansion(vmin=-1.0, vmax=4.0, bins=nbasis)

    def forward(self, r):
        curve = (self.weights * self.basis(r)).sum(dim=1)
        # subtract sum of parameters to shift the curve so f(0) = 0
        # z = self.weights * self.basis(torch.tensor(0).unsqueeze(1))
        return curve


class EmbeddedAtomPotential(torch.nn.Module):
    def __init__(self, nbasis: int = 128, cutoff: float = 6.0):
        super().__init__()

        self.nbasis = nbasis
        self.cutoff = cutoff

        self.density = GaussianSpline(
            nbasis=nbasis, cutoff=cutoff, activation="softplus"
        )
        # self.pair_repulsion = ExponentialRepulsive()
        self.pair_repulsion = GaussianSpline(nbasis=nbasis, cutoff=cutoff)
        self.embedding_energy = PolynomialEmbeddingFunction()

        self.transform = nfflr.nn.PeriodicRadiusGraph(self.cutoff)

    def reset_parameters(self):
        torch.nn.init.normal_(self.density.phi.data, -1.0, 0.1)
        # phis = 1 / (1 + self.pair_repulsion.basis.centers)
        self.pair_repulsion.phi.data = 10 * torch.exp(
            -0.1 * self.pair_repulsion.basis.centers
        )
        # torch.nn.init.normal_(self.pair_repulsion.phi.data, phis, 0.1)
        # torch.nn.init.normal_(self.embedding_energy.weights.data, 0.0, 0.5)

    def forward(self, at: nfflr.Atoms):
        if type(at) == nfflr.Atoms:
            g = self.transform(at)
        else:
            g = at

        g = g.local_var()

        # initial bond features: bond displacement vectors
        # need to add bond vectors to autograd graph
        r = g.edata.pop("r")
        r.requires_grad_(True)

        bondlen = torch.norm(r, dim=1)

        g.edata["density_ij"] = self.density(bondlen)
        g.update_all(fn.copy_e("density_ij", "m"), fn.sum("m", "local_density"))
        g.ndata["F"] = self.embedding_energy(g.ndata["local_density"])
        g.edata["pair_repulsion"] = self.pair_repulsion(bondlen) / bondlen
        potential_energy = dgl.readout_nodes(g, "F") + dgl.readout_edges(
            g, "pair_repulsion"
        )

        # potential_energy = F.sum() + pair_repulsion.sum()

        forces = nfflr.autograd_forces(potential_energy, r, g, energy_units="eV")
        return {"total_energy": potential_energy, "forces": forces}


class ExponentialDensity(torch.nn.Module):
    def __init__(self, amplitude: float = 3.0, lengthscale: float = 1.5):
        super().__init__()

        def inv_softplus(r):
            return r + np.log(-np.expm1(-r))

        self.amplitude = torch.nn.Parameter(torch.tensor(inv_softplus(amplitude)))
        self.lengthscale = torch.nn.Parameter(torch.tensor(inv_softplus(lengthscale)))

    def forward(self, r):
        return softplus(self.amplitude) * torch.exp(-softplus(self.lengthscale) * r)


class SqrtEmbedding(torch.nn.Module):
    def __init__(self, amplitude: float = 3.0, lengthscale: float = 1.0):
        super().__init__()

        def inv_softplus(r):
            return r + np.log(-np.expm1(-r))

        self.amplitude = torch.nn.Parameter(torch.tensor(inv_softplus(amplitude)))
        # self.lengthscale = torch.nn.Parameter(torch.tensor(inv_softplus(lengthscale)))

    def forward(self, density):
        return -softplus(self.amplitude) * torch.sqrt(density)


class SimpleEmbeddedAtomPotential(torch.nn.Module):
    def __init__(self, cutoff: float = 6.0):
        super().__init__()
        self.cutoff = cutoff
        self.transform = nfflr.nn.PeriodicRadiusGraph(self.cutoff)

        # self.density = ElectronDensity(nbasis=128, cutoff=cutoff)
        self.density = ExponentialDensity()
        self.pair_repulsion = ExponentialRepulsive(amplitude=10.0)
        self.embedding_energy = SqrtEmbedding()

    def forward(self, at: nfflr.Atoms):
        if type(at) == nfflr.Atoms:
            g = self.transform(at)
        else:
            g = at

        g = g.local_var()

        # initial bond features: bond displacement vectors
        # need to add bond vectors to autograd graph
        r = g.edata.pop("r")
        r.requires_grad_(True)

        bondlen = torch.norm(r, dim=1)

        g.edata["density_ij"] = self.density(bondlen)
        g.update_all(fn.copy_e("density_ij", "m"), fn.sum("m", "local_density"))
        g.ndata["F"] = self.embedding_energy(g.ndata["local_density"])
        g.edata["pair_repulsion"] = self.pair_repulsion(bondlen)  # / bondlen

        potential_energy = dgl.readout_nodes(g, "F") + dgl.readout_edges(
            g, "pair_repulsion"
        )
        # potential_energy = F.sum() + pair_repulsion.sum()

        forces = nfflr.autograd_forces(potential_energy, r, g, energy_units="eV")
        return {"total_energy": potential_energy, "forces": forces}

        return
