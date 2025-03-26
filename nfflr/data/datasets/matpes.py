import gzip
import json
from pathlib import Path
from typing import Literal

import ase
import ase.db
import numpy as np

from nfflr.data.asedataset import AtomsSQLDataset


def pmg_to_ase(atoms: dict):
    """load atoms from pymatgen dict without pymatgen dependency."""
    # ignores partially occupied sites...
    cell = atoms["lattice"]["matrix"]
    coords = np.asarray([site["xyz"] for site in atoms["sites"]])
    symbols = [site["species"][0]["element"] for site in atoms["sites"]]
    magmoms = [site["properties"]["magmom"] for site in atoms["sites"]]
    return ase.Atoms(cell=cell, positions=coords, symbols=symbols, magmoms=magmoms)


def load_reference_calcs(
    dataset_dir: Path = Path("."), functional: Literal["PBE", "R2SCAN"] = "PBE"
):
    """Load isolated atom reference calculations."""

    with gzip.open(dataset_dir / f"MatPES-{functional}-2025.1.json.gz", "r") as f:
        records = json.load(f)

    with gzip.open(dataset_dir / f"MatPES-{functional}-atoms.json.gz", "r") as f:
        ref = json.load(f)

    with gzip.open(dataset_dir / f"MatPES-{functional}-split.json.gz", "r") as f:
        split = json.load(f)

    return records, ref, split


def json_to_sql(
    dataset_dir: Path = Path("."), functional: Literal["PBE", "R2SCAN"] = "PBE"
):
    dbpath = dataset_dir / f"MatPES-{functional}-2025.1.db"
    records, ref, split_ids = load_reference_calcs(dataset_dir, functional)

    split = np.repeat(["train"], len(records))
    split[split_ids["valid"]] = "val"
    split[split_ids["test"]] = "test"
    print(np.unique(split))

    # atomic_number -> atomic_energy
    atomic_energies = {
        ase.data.atomic_numbers[item["elements"][0]]: item["energy"] for item in ref
    }

    with ase.db.connect(dbpath) as db:
        db.metadata = {"atomic_energies": atomic_energies}

        for idx, record in enumerate(records):
            atoms = pmg_to_ase(record["structure"])
            atoms.calc = ase.calculators.singlepoint.SinglePointCalculator(atoms=atoms)
            atoms.calc.results["energy"] = record["energy"]
            atoms.calc.results["forces"] = np.asarray(record["forces"])

            # stress is in kbar (vasp sign convention?) according to https://matpes.ai/dataset
            # according to
            # https://github.com/materialsvirtuallab/matpes/blob/main/notebooks/Training%20a%20MatPES%20model.ipynb
            # the stress is stored in ase voigt_6 format, but in kbar and vasp sign convention?
            stress = -0.1 * ase.units.GPa * np.asarray(record["stress"])
            atoms.calc.results["stress"] = stress

            rowdata = dict(
                atoms=atoms,
                frame_id=record["matpes_id"],
                matpes_index=idx,
                split=split[idx],
                mp_id=record["provenance"]["original_mp_id"],
                md_step=record["provenance"].get("md_step"),
            )

            if rowdata["md_step"] is None:
                del rowdata["md_step"]

            db.write(**rowdata)


def matpes_dataset(
    functional=Literal["PBE", "R2SCAN"],
    cohesive_energies: bool = True,
    dbpath: Path = Path("."),
    **kwargs,
):

    return AtomsSQLDataset(
        "./MatPES-PBE-2025.1.db",
        cohesive_energies=cohesive_energies,
        train_val_seed="predefined",
        **kwargs,
    )
