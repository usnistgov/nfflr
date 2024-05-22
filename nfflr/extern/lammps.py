"""LAMMPS MLIAP_UNIFIED interface."""
from typing import TypeAlias

import numpy as np
from lammps.mliap.mliap_unified_abc import MLIAPUnified

# forward reference for dynamically linked MLIAPData type
""" https://github.com/lammps/lammps/blob/develop/src/ML-IAP/mliap_data.h#L21"""
MLIAPDataPy: TypeAlias = "mliap_unified_couple.MLIAPDataPy"  # noqa:F821


class MLIAPModel(MLIAPUnified):
    """Test implementation for MLIAPUnified."""

    def __init__(self, element_types, epsilon=1.0, sigma=1.0, rcutfac=1.25):
        # extra MLIAP arguments not needed for MLIAPUnified model
        _interface = None
        _ndescriptors = 1
        _nparams = 3
        # ARGS: interface, element_types, ndescriptors, nparams, rcutfac
        super().__init__(_interface, element_types, _ndescriptors, _nparams, rcutfac)

        # Mimicking the LJ pair-style:
        # pair_style lj/cut 2.5
        # pair_coeff * * 1 1
        self.epsilon = epsilon
        self.sigma = sigma

    def compute_gradients(self, data: MLIAPDataPy):
        """Test compute_gradients."""

    def compute_descriptors(self, data: MLIAPDataPy):
        """Test compute_descriptors."""

    def compute_forces(self, data: MLIAPDataPy):
        """Test compute_forces."""
        eij, fij = self.compute_pair_ef(data)
        data.update_pair_energy(eij)
        data.update_pair_forces(fij)

    def compute_pair_ef(self, data: MLIAPDataPy):
        # print(dir(data))
        print(f"{data.rij.shape=}")

        # cell
        # lmp = self.lmp
        # lmp_np = self.lmp.numpy

        # print(f"{lmp.extract_box()=}")

        # neighborlist in iatoms, jatoms
        src = data.iatoms
        dst = data.jatoms

        # # atom ids
        # print(f"{lmp_np.extract_atom('id')=}")
        # # atom positions
        # print(f"{lmp_np.extract_atom('x')=}")
        # # atom types
        # print(f"{lmp_np.extract_atom('type')=}")

        rij = data.rij

        # possibly use `compute angle/local` to get triplets fast?
        # or maybe `compute property/local` https://docs.lammps.org/compute_property_local.html
        # print(f"{lmp_np.gather_angles()=}")

        r2inv = 1.0 / np.sum(rij**2, axis=1)
        r6inv = r2inv * r2inv * r2inv

        lj1 = 4.0 * self.epsilon * self.sigma**12
        lj2 = 4.0 * self.epsilon * self.sigma**6

        eij = r6inv * (lj1 * r6inv - lj2)
        fij = r6inv * (3.0 * lj2 - 6.0 * lj2 * r6inv) * r2inv
        fij = fij[:, np.newaxis] * rij
        return eij, fij
