__all__ = ()

from pathlib import Path
from typing import Optional

import torch
import jarvis
import numpy as np
import pandas as pd
from cached_path import cached_path

import nfflr


def pmg_to_nfflr(atoms: dict):
    """load atoms from pymatgen dict without pymatgen dependency."""
    # ignores partially occupied sites...
    cell = atoms["lattice"]["matrix"]
    coords = torch.tensor([site["abc"] for site in atoms["sites"]])
    symbols = [site["species"][0]["element"] for site in atoms["sites"]]
    numbers = [jarvis.core.specie.chem_data[sym]["Z"] for sym in symbols]

    return nfflr.Atoms(cell, coords, numbers)


def _mlearn_dataset(datafile: Path):
    """pymatgen distribution of mlearn dataset."""
    data = pd.read_json(datafile)

    data["atoms"] = data.structure.apply(pmg_to_nfflr)
    data["energy"] = data.outputs.apply(lambda x: x["energy"])
    data["forces"] = data.outputs.apply(lambda x: x["forces"])
    data["stresses"] = data.outputs.apply(lambda x: x["virial_stress"])
    data["jid"] = data.index

    return nfflr.AtomsDataset(data, target="energy_and_forces", energy_units="eV")


def mlearn_dataset(
    elements: str | list[str] = "Si",
    transform: Optional[torch.nn.Module] = None,
    diskcache: bool = False,
):
    """Construct mlearn dataset with standard splits.

    Downloads and caches json datafiles from github.com/materialsvirtuallab/mlearn
    to nfflr.CACHE directory, which respects `XDG_CACHE_HOME`.
    """
    mlearn_base = "https://github.com/materialsvirtuallab/mlearn/raw/master/data"

    if isinstance(elements, str):
        elements = [elements]

    datafiles = []
    for element in elements:
        datafiles.append(f"{mlearn_base}/{element}/training.json")
        datafiles.append(f"{mlearn_base}/{element}/test.json")

    dfs = [
        pd.read_json(cached_path(datafile, cache_dir=nfflr.CACHE))
        for datafile in datafiles
    ]
    df = pd.concat(dfs, ignore_index=True)

    df["atoms"] = df.structure.apply(pmg_to_nfflr)
    df["energy"] = df.outputs.apply(lambda x: x["energy"])
    df["forces"] = df.outputs.apply(lambda x: x["forces"])
    df["stresses"] = df.outputs.apply(lambda x: x["virial_stress"])
    df["jid"] = df.index

    dataset = nfflr.AtomsDataset(
        df,
        target="energy_and_forces",
        energy_units="eV",
        transform=transform,
        diskcache=diskcache,
    )

    # use the standard split: override random splits
    (id_train,) = np.where(df.tag == "train")
    (id_val,) = np.where(df.tag == "test")
    id_test = np.array([])
    dataset.split = dict(train=id_train, val=id_val, test=id_test)

    # redo target standardization since it requires the correct train split
    if dataset.standardize:
        dataset.setup_target_standardization()

    return dataset
