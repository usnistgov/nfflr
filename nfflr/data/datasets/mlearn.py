__all__ = ()

from pathlib import Path
from typing import Optional

import torch
import jarvis
import numpy as np
import pandas as pd
from einops import rearrange

import nfflr


def pmg_to_nfflr(atoms: dict):
    """load atoms from pymatgen dict without pymatgen dependency."""
    # ignores partially occupied sites...
    cell = atoms["lattice"]["matrix"]
    coords = torch.tensor([site["abc"] for site in atoms["sites"]])
    symbols = [site["species"][0]["element"] for site in atoms["sites"]]
    numbers = [jarvis.core.specie.chem_data[sym]["Z"] for sym in symbols]

    return nfflr.Atoms(cell, coords, numbers)


def mlearn_dataset(datafile: Path):
    """pymatgen distribution of mlearn dataset."""
    data = pd.read_json(datafile)

    data["atoms"] = data.structure.apply(pmg_to_nfflr)
    data["energy"] = data.outputs.apply(lambda x: x["energy"])
    data["forces"] = data.outputs.apply(lambda x: x["forces"])
    data["stresses"] = data.outputs.apply(lambda x: x["virial_stress"])
    data["jid"] = data.index

    return nfflr.AtomsDataset(data, target="energy_and_forces", energy_units="eV")
