from typing import Callable, Optional, Literal
from pathlib import Path

import json
import shutil
import warnings
import contextlib
import sqlite3

import torch
import numpy as np
import pandas as pd
from numpy.random import default_rng

import ase
import ase.db

import lmdb
import pickle

import nfflr
from nfflr.data.dataset import to_tensor, get_cachedir


class AtomsSQLDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dbpath: Path | str,
        transform: Optional[Callable] = None,
        cohesive_energies: bool = False,
        n_train: float | int = 0.9,
        n_val: float | int = 0.1,
        group_ids: bool = True,
        train_val_seed: int = 42,
        split: Literal["predefined"] | str | Path | None = None,
        diskcache: Optional[Path | str] = None,
        use_lmdb: bool = False,
        cache_structures: bool = True,
        copy_db_to_scratch: bool = False,
    ):
        super().__init__()

        if isinstance(dbpath, str):
            dbpath = Path(dbpath)

        self.dbpath = dbpath
        self.transform = transform
        self.group_ids = group_ids

        self.cache_structures = cache_structures
        if diskcache is not None and diskcache is not False:
            # TemporaryDirectory
            self.cachedir = get_cachedir(diskcache)
            print(self.cachedir)
            if use_lmdb:
                self.diskcache = lmdb.open(self.cachedir.name, map_size=1024**4)
            else:
                self.diskcache = self.cachedir

            if copy_db_to_scratch:
                # replicate the database on scratch storage and load from there
                shutil.copy2(dbpath, Path(self.cachedir.name) / dbpath.name)
                self.dbpath = Path(self.cachedir.name) / dbpath.name

        else:
            self.diskcache = None

        self.collate = nfflr.AtomsDataset.collate_forcefield
        self.prepare_batch = nfflr.AtomsDataset.prepare_batch_default

        self.train_val_seed = None
        if split is None:
            self.train_val_seed = train_val_seed
            # loading the dataset splits takes a bit for larger datasets...
            self.split = self.split_dataset_by_id(n_train, n_val)

        elif split == "predefined":
            self.split = self.predefined_split()

        else:
            if Path(split).is_file():
                with open(split, "r") as f:
                    splitdata = json.load(f)
            self.split = {key: np.asarray(val) for key, val in splitdata.items()}

        if cohesive_energies:
            with ase.db.connect(self.dbpath) as db:
                # stored as dict[str,float]
                self.atomic_energies = db.metadata.get("atomic_energies")
            if self.atomic_energies is None:
                raise ValueError(
                    "database must contain atomic reference energies"
                    "to compute cohesive energies."
                )

            # convert string keys to integer
            self.atomic_energies = {
                int(key): value for key, value in self.atomic_energies.items()
            }

        self.cohesive_energies = cohesive_energies

    def __len__(self):
        with ase.db.connect(self.dbpath) as db:
            sz = len(db)

        return sz

    def load_atomsrow(self, idx: int):

        with ase.db.connect(self.dbpath) as db:
            # dataset index zero-based
            # sqlite primary keys start at one
            row = db.get(idx + 1)
        return row

    def _load_atoms(self, key: str):

        result = None
        if isinstance(self.diskcache, lmdb.Environment):
            with self.diskcache.begin() as tx:
                result = tx.get(key.encode())

            if result is not None:
                result = pickle.loads(result)

        else:
            cachefile = Path(self.diskcache.name) / f"{key}.pkl"
            if cachefile.is_file():
                result = torch.load(cachefile)

        return result

    def _cache_atoms(self, key: str, atoms):

        if isinstance(self.diskcache, lmdb.Environment):
            with self.diskcache.begin(write=True) as tx:
                tx.put(key.encode(), pickle.dumps(atoms))
        else:
            cachefile = Path(self.diskcache.name) / f"{key}.pkl"
            torch.save(atoms, cachefile)

    def produce_or_load_atoms(self, row: ase.db.row.AtomsRow, idx: int):

        frame_id = row.frame_id.replace("/", "_")
        key = f"{frame_id}-{idx}"

        if self.cache_structures and self.diskcache is not None:
            atoms = self._load_atoms(key)
            if atoms is not None:
                return atoms

        atoms = nfflr.Atoms(row.cell, row.positions, row.numbers)

        if self.transform is not None:
            atoms = self.transform(atoms)

        if self.cache_structures and self.diskcache is not None:
            self._cache_atoms(key, atoms)

        return atoms

    def __getitem__(self, idx: int):
        # print(f"{idx=}")

        row = self.load_atomsrow(int(idx))
        atoms = self.produce_or_load_atoms(row, idx)
        # key = row.frame_id

        # NOTE: careful loading row.stress, ase symmetrizes the stress tensor...
        if len(row.stress) == 9:
            stress = row.stress.reshape(3, 3)
        elif len(row.stress) == 6:
            stress = ase.stress.voigt_6_to_full_3x3_stress(row.stress)

        refs = dict(
            energy=to_tensor(row.energy),
            forces=to_tensor(row.forces),
            stress=to_tensor(stress),
            volume=row.volume,
            idx=idx,
        )
        refs["virial"] = -row.volume * refs["stress"]

        if self.cohesive_energies:
            # subtract off atomic reference energies to obtain cohesive energy
            refs["energy"] -= sum(self.atomic_energies[z] for z in row.numbers)

        return atoms, refs

    def predefined_split(self):
        """Load train/val/test splits from text metadata 'split'."""

        # dict[Literal["train", "val", "test"]: Int[np.ndarray, "split_size"]]
        with contextlib.closing(sqlite3.connect(self.dbpath)) as connection:
            # load the entire group_id column from ase sqlite database
            # directly run sql query from pandas instead of going through ase
            splits = pd.read_sql(
                "select * from text_key_values where key='split'", connection
            )
            # sort by ase.db primary key
            splits = splits.sort_values(by="id")
            split_ids = splits["value"].values

        split_keys = np.unique(split_ids)
        return {key: np.where(split_ids == key)[0] for key in split_keys}

    def elemental_subset(self, elements: set | list):
        """filter dataset splits to contain only subsets of `elements`."""
        elements = set(elements)
        with ase.db.connect(self.dbpath) as db:
            subset = lambda row: set(row.symbols).issubset(elements)
            ids = [row.id for row in db.select(filter=subset)]

        split = {
            key: np.asarray([_id for _id in ids if _id in self.split[key]])
            for key in self.split.keys()
        }

        return split

    def split_dataset_by_id(self, n_train: float | int, n_val: float | int):
        """Get train/val/test split indices for SubsetRandomSampler.

        Stratify by calculation / trajectory id `"group_id"`
        """
        if self.group_ids:
            with contextlib.closing(sqlite3.connect(self.dbpath)) as connection:
                # load the entire group_id column from ase sqlite database
                # directly run sql query from pandas instead of going through ase
                groups = pd.read_sql(
                    "select * from text_key_values where key='group_id'", connection
                )
                # sort by ase.db primary key
                groups = groups.sort_values(by="id")
                all_ids = groups["value"].values
        else:
            all_ids = np.arange(len(self))

        unique_ids = np.unique(all_ids)

        # number of calculation groups
        N = unique_ids.size

        if isinstance(n_train, float) and isinstance(n_val, float):
            # n_train and n_val specified as fractions of dataset
            if n_train + n_val > 1.0:
                warnings.warn("training and validation set exceed 100% of data")
            n_val = int(n_val * N)
            n_train = int(n_train * N)

        if isinstance(n_train, int) and isinstance(n_val, int):
            # n_train and n_val specified directly as calculation group counts
            if n_train + n_val > N:
                warnings.warn("training and validation set exceed 100% of data")

        # shuffle in place with configurable train/val seed
        if self.train_val_seed != "predefined":
            train_val_rng = default_rng(self.train_val_seed)
        train_val_rng.shuffle(unique_ids)

        split_ids = {
            "val": unique_ids[0:n_val],
            "train": unique_ids[n_val : n_val + n_train],
        }

        # split atomistic configurations
        # store the index as a column,
        # set the string key to the pandas index,
        # then slice with loc
        groups = pd.DataFrame(dict(id=all_ids))
        groups["idx"] = groups.index
        groups.set_index("id", inplace=True)

        # e.g. groups.loc[split_ids["train"]
        splitdata = {
            split_key: groups.loc[_split_ids]["idx"].values
            for split_key, _split_ids in split_ids.items()
        }

        return splitdata


# dataset = AtomsSQLDataset(datadir.parent / "mptrj_train.db")
