"""NFFLr - neural force field learning toolkit."""
__version__ = "0.1.1"

__all__ = ["Atoms", "AtomsDataset"]

# nfflr: top-level API
# Atoms
from .data.atoms import Atoms, spglib_cell, batch, unbatch, to_ase

# AtomsDataset
from .data.dataset import AtomsDataset, collate_forcefield_targets

# utilities
from .models.utils import autograd_forces

# modeling primitives
from . import nn

# models - classical and neural network
from . import models

# datasets
from . import data

# ASE interface
from .ase import NFFLrCalculator
