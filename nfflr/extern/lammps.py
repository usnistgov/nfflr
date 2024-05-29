"""LAMMPS MLIAP_UNIFIED interface."""
from typing import TypeAlias

import dgl
import torch
import numpy as np
from lammps.mliap.mliap_unified_abc import MLIAPUnified

# forward reference for dynamically linked MLIAPData type
""" https://github.com/lammps/lammps/blob/develop/src/ML-IAP/mliap_data.h#L21"""
MLIAPDataPy: TypeAlias = "mliap_unified_couple.MLIAPDataPy"  # noqa:F821


class MLIAPModel(MLIAPUnified):
    """Test implementation for MLIAPUnified."""

    def __init__(
        self,
        element_types: list[str],
        base_model: torch.nn.Module,
        rcutfac=1.25,
        lmp=None,
    ):
        # extra MLIAP arguments not needed for MLIAPUnified model
        _interface = None
        _ndescriptors = 1
        _nparams = 3
        # ARGS: interface, element_types, ndescriptors, nparams, rcutfac
        super().__init__(_interface, element_types, _ndescriptors, _nparams, rcutfac)

        self.dtype = torch.get_default_dtype()

        # hack - use lammps library interface to extract cell and positions
        self.lmp = lmp

        # Mimicking the LJ pair-style:
        # pair_style lj/cut 2.5
        # pair_coeff * * 1 1
        self.base_model = base_model

        self.compute_pair_ef = self.compute_pair_ef_mliap

    def compute_gradients(self, data: MLIAPDataPy):
        """Test compute_gradients."""

    def compute_descriptors(self, data: MLIAPDataPy):
        """Test compute_descriptors."""

    def compute_forces(self, data: MLIAPDataPy):
        """Test compute_forces."""
        eij, fij = self.compute_pair_ef(data)
        data.update_pair_energy(eij)
        data.update_pair_forces(fij)

    def compute_pair_ef_shlib(self, data: MLIAPDataPy):
        print("start compute_pair_ef_shlib")

        # cell
        lmp = self.lmp
        lmp_np = self.lmp.numpy

        # assuming no spatial decomposition, can evaluate energy+forces
        # however: still need to resolve ghost atom ids to pack pair forces!
        # possible fix: extend mliap_data.cpp generate_neighdata with atom->tag
        # which should be the global atom ID
        # enabling direct construction of dgl.DGLGraph from neighborlist data
        # see https://docs.lammps.org/Developer_atom.html#atom-indexing
        # https://github.com/lammps/lammps/blob/develop/src/ML-IAP/mliap_data.cpp#L108
        box = lmp.extract_box()
        ids = torch.from_numpy(lmp_np.extract_atom("id"))
        xs = torch.from_numpy(lmp_np.extract_atom("x"))
        types = torch.from_numpy(lmp_np.extract_atom("type"))

        print(f"{box=}")
        print(f"{ids=}")
        print(f"{xs.shape=}, {xs=}")
        print(f"{types=}")

        # neighborlist in iatoms, jatoms
        # jatoms seems to include ghost atoms
        # that have different ids than their periodic images...
        i = torch.from_numpy(data.iatoms)
        # tags = torch.from_numpy(data.itags)
        # print(f"{tags=}")
        numneighs = torch.from_numpy(data.numneighs)
        dst = torch.repeat_interleave(i, numneighs)

        src = torch.from_numpy(data.jatoms)
        print(f"{dst.shape=}, {src.shape=}")

        print(f"{src.unique().size()=}")
        print(f"{dst.unique().size()=}")

        g = dgl.graph((src, dst))
        print("graph created.")
        g.ndata["atomic_number"] = torch.from_numpy(data.ielems)
        print("zs assigned")
        g.edata["r"] = torch.from_numpy(data.rij)

        results = self.base_model(g)

        rij = data.rij

        r2inv = 1.0 / np.sum(rij**2, axis=1)
        r6inv = r2inv * r2inv * r2inv

        lj1 = 4.0 * self.epsilon * self.sigma**12
        lj2 = 4.0 * self.epsilon * self.sigma**6

        eij = r6inv * (lj1 * r6inv - lj2)
        fij = r6inv * (3.0 * lj2 - 6.0 * lj2 * r6inv) * r2inv
        fij = fij[:, np.newaxis] * rij
        return eij, fij

    def compute_pair_ef_mliap(self, data: MLIAPDataPy):
        """Compute pair energy and forces with pytorch model.

        Construct dgl.DGLGraph directly from LAMMPS neighborlist.
        Limited to serial computation (on the LAMMPS side due to comms).
        """
        # load neighborlist using tags (LAMMPS global atom ids)
        # jtags includes ghost atoms which have local indices (jatoms)
        # that don't match their periodic images (iatoms)
        i = torch.from_numpy(data.itags).type(torch.int)
        numneighs = torch.from_numpy(data.numneighs).type(torch.int)
        dst = torch.repeat_interleave(i, numneighs)

        src = torch.from_numpy(data.jtags).type(torch.int)

        # LAMMPS atom IDS start from 1
        g = dgl.graph((src - 1, dst - 1))
        g.ndata["atomic_number"] = torch.from_numpy(data.ielems).type(torch.int)
        g.edata["r"] = torch.from_numpy(data.rij).type(self.dtype)
        results = self.base_model(g)

        # this should not use force reduction, LAMMPS wants pairwise forces
        # the factor of 2 is because the GNN forces are not symmetrized
        # and MLIAP internally multiplies by 1/2
        fij = 2 * results["forces"].type(torch.float64)
        eij = (
            results["energy"]
            / fij.shape[0]
            * torch.ones(fij.shape[0], dtype=torch.float64)
        )

        return eij.detach().numpy(), fij.detach().numpy()
