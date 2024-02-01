"""Dataset construction based on collections of Vasp outputs."""

from pathlib import Path
from typing import Iterator, Optional

import ase
import torch
import numpy as np
import pandas as pd
from jarvis.io.vasp.outputs import Vasprun

import nfflr


def _get_key(collection, key):
    """Indirect attribute access for vasprun.xml key/value stores."""
    varray = collection["varray"]

    if isinstance(varray, dict):
        varray = [varray]

    for item in varray:
        if item["@name"] == key:
            return item["v"]


def get_energy(step, key="e_fr_energy"):
    """Read energy from vasprun.xml ionic step data."""
    for entry in step["energy"]["i"]:
        if entry["@name"] == key:
            return float(entry["#text"])


def parse_array(vs: list[str]) -> np.array:
    """Parse vasprun.xml arrays into numpy array"""
    return np.array([list(map(float, v.split())) for v in vs])


def getcell(step):
    """Parse cell and atom positions from vasprun.xml ionic step data."""
    s = step["structure"]
    cell = parse_array(_get_key(s["crystal"], "basis"))
    xs = parse_array(_get_key(s, "positions"))

    return dict(cell=cell, positions=xs)


def load_vasp_steps(path: Path, dataset_dir: Path):
    """Load data for all ionic steps in vasprun.xml

    path: full path to `vasprun.xml`
    dataset_dir: top-level data directory used to determine group ids
    """

    group_id = str(path.relative_to(dataset_dir).parent)

    v = Vasprun(path)
    symbols = v.elements

    configurations = []

    for step_id, step in enumerate(v.ionic_steps):
        data = dict(
            id=f"{group_id}.{step_id}",
            group_id=group_id,
            step_id=step_id,
            symbols=symbols,
        )
        data.update(getcell(step))

        data["energy"] = get_energy(step)
        data["forces"] = parse_array(_get_key(step, "forces"))
        data["stress"] = parse_array(_get_key(step, "stress"))

        configurations.append(data)

    return configurations


def load_vasp_steps_jv(path: Path, dataset_dir: Path):
    """Load data for all ionic steps in vasprun.xml

    path: full path to `vasprun.xml`
    dataset_dir: top-level data directory used to determine group ids
    """

    group_id = str(path.relative_to(dataset_dir).parent)
    v = Vasprun(path)

    configurations = []
    data = zip(v.all_structures, v.all_energies, v.all_forces, v.all_stresses)
    for step_id, (atoms, energy, forces, stress) in enumerate(data):
        data = dict(
            id=f"{group_id}.{step_id}",
            group_id=group_id,
            step_id=step_id,
            atoms=atoms.to_dict(),
            energy=energy,
            forces=forces,
            stress=stress,
        )
        configurations.append(data)

    return configurations


def vasprun_dataset(
    datafiles: Iterator[Path],
    datadir: Path,
    transform: Optional[torch.nn.Module] = None,
    diskcache: bool = False,
):
    """Construct an AtomsDataset from a collection of vasprun.xml files.

    e.g. datafiles = datadir.glob("*/*/vasprun.xml")
    """
    data = []
    for datafile in datadir.glob("*/*/vasprun.xml"):
        try:
            data += load_vasp_steps_jv(datafile, datadir)
        except:
            print(f"error loading {datafile}")

    return nfflr.AtomsDataset(
        pd.DataFrame(data),
        id_tag="id",
        target="energy_and_forces",
        energy_units="eV",
        transform=transform,
        diskcache=diskcache,
    )
