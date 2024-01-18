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
    lattice = atoms["lattice"]["matrix"]
    coords = torch.tensor([site["abc"] for site in atoms["sites"]])
    symbols = [site["species"][0]["element"] for site in atoms["sites"]]
    numbers = [jarvis.core.specie.chem_data[sym]["Z"] for sym in symbols]

    return nfflr.Atoms(lattice, coords, numbers)


def mlearn_dataset(datafile: Path):
    """pymatgen distribution of mlearn dataset."""
    data = pd.read_json(datafile)

    data["atoms"] = data.structure.apply(pmg_to_nfflr)
    data["energy"] = data.outputs.apply(lambda x: x["energy"])
    data["forces"] = data.outputs.apply(lambda x: x["forces"])
    data["stresses"] = data.outputs.apply(lambda x: x["virial_stress"])
    data["jid"] = data.index

    return nfflr.AtomsDataset(data, target="energy_and_forces", energy_units="eV")


def deepmd_load_atomic_numbers(datadir: Path):
    """Load atomic species from deepmd raw format."""
    # all frames in a folder have the same atom id -> species mapping
    species = np.loadtxt(datadir / "type_map.raw", dtype=str)
    ids = np.loadtxt(datadir / "type.raw", dtype=int)
    numbers = np.array([jarvis.core.specie.chem_data[sym]["Z"] for sym in species[ids]])

    return numbers


def deepmd_load_frameset(path, zs):
    """Load MD frames from deepmd numpy format."""
    box = torch.tensor(np.load(path / "box.npy"))
    # energies stored as meV I think
    energy = torch.tensor(np.load(path / "energy.npy")) / 1000
    coord = torch.tensor(np.load(path / "coord.npy"))
    force = torch.tensor(np.load(path / "force.npy"))

    box = rearrange(box, "cells (dim1 dim2) -> cells dim1 dim2", dim1=3)
    coord = rearrange(coord, "cells (atoms dims) -> cells atoms dims", dims=3)
    force = rearrange(force, "cells (atoms dims) -> cells atoms dims", dims=3)

    at = [nfflr.Atoms(b, x @ np.linalg.inv(b), zs) for b, x in zip(box, coord)]

    return at, energy, force.unbind(0)


def deepmd_hea_dataset(
    datadir: Path | str, transform: Optional[torch.nn.Module] = None
):
    """Load deepmd numpy format dataset."""
    # only supports HEA dataset for now...
    # training data are under rand1, set00{0..8}
    # validation data are under rand1, set009
    # test data are under rand2

    # train/val data

    if isinstance(datadir, str):
        datadir = Path(datadir)

    data = []
    for set_dir in (datadir / "rand1").glob("*/set*"):
        key = ".".join(set_dir.parts[-3:])
        zs = deepmd_load_atomic_numbers(set_dir.parent)
        at, e, f = deepmd_load_frameset(set_dir, zs)
        jid = [f"{key}_{idx}" for idx in range(len(at))]
        data.append({"atoms": at, "total_energy": e, "forces": f, "jid": jid})

    df = pd.concat([pd.DataFrame(_data) for _data in data])

    return nfflr.AtomsDataset(
        df,
        target="energy_and_forces",
        energy_units="eV",
        group_ids=True,
        n_train=0.9,
        n_val=0.1,
        transform=transform,
    )
