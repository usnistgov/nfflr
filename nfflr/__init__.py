"""NFFLr - neural force field learning toolkit."""
# ruff: noqa: F401
__version__ = "0.4.1"

__all__ = ["Atoms", "AtomsDataset", "CACHE"]

import warnings

# nfflr: top-level API
# Atoms
from .atoms import Atoms, spglib_cell, batch, unbatch, to_ase

# AtomsDataset
from .data.dataset import AtomsDataset, collate_forcefield_targets
from .data.asedataset import AtomsSQLDataset


try:
    # utilities
    from .models.utils import autograd_forces

    # modeling primitives
    from . import nn

    # models - classical and neural network
    from . import models

except ImportError:
    warnings.warn("DGL not found - not loading modeling components.")

# datasets
from . import data

# training
from . import train

# ASE interface
from .ase import NFFLrCalculator

# cache directory
from pathlib import Path
import platformdirs

CACHE = platformdirs.user_cache_dir("nfflr", "nfflr")
