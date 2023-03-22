"""Standalone dataset for training force field models."""
import os
import tempfile
from functools import partial
from typing import List, Literal, Tuple, Union, Optional, Callable

import dgl
import numpy as np
import pandas as pd
import torch
from jarvis.core.atoms import Atoms as jAtoms
from numpy.random import default_rng

from nfflr.atoms import Atoms


def get_cachedir():
    """Get local scratch directory."""

    scratchdir = None
    prefix = "nfflr-"

    slurm_job = os.environ.get("SLURM_JOB_ID")
    if slurm_job is not None:
        scratchdir = "/scratch"
        prefix = f"{slurm_job}-"

    cachedir = tempfile.TemporaryDirectory(dir=scratchdir, prefix=prefix)

    return cachedir


class AtomsDataset(torch.utils.data.Dataset):
    """Dataset of Atoms."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str = "formation_energy_peratom",
        transform: Optional[Callable] = None,
        train_val_seed: int = 42,
        id_tag: str = "jid",
        n_train: Union[float, int] = 0.8,
        n_val: Union[float, int] = 0.1,
        energy_units: Literal["eV", "eV/atom"] = "eV/atom",
        diskcache: bool = False,
    ):
        """Pytorch Dataset for atomistic graphs.

        `df`: pandas dataframe from e.g. jarvis.db.figshare.data
        """
        example_id = df[id_tag].iloc[0]

        if isinstance(example_id, int):
            df["group_id"] = df[id_tag]
            df["step_id"] = df.index
        elif "_" not in example_id:
            df["group_id"] = df[id_tag]
            df["step_id"] = np.ones(df.shape[0])
        else:
            # split key like "JVASP-6664_main-5"
            df["group_id"], df["step_id"] = zip(
                *df[id_tag].apply(partial(str.split, sep="_"))
            )

        self.atoms = df.atoms.apply(lambda x: Atoms(jAtoms.from_dict(x)))
        self.transform = transform
        self.target = target

        self.df = df
        self.train_val_seed = train_val_seed
        self.id_tag = id_tag
        self.energy_units = energy_units
        self.ids = self.df[id_tag]

        self.split = self.split_dataset_by_id(n_train, n_val)

        if diskcache:
            self.diskcache = get_cachedir()
        else:
            self.diskcache = None

    def __len__(self):
        """Get length."""
        return self.df.shape[0]

    def __getitem__(self, idx):
        """Get AtomsDataset sample."""

        key = self.df[self.id_tag].iloc[idx]
        atoms = self.atoms[idx]

        if self.diskcache is not None:
            cachefile = self.diskcache / f"jarvis-{key}.pkl"

            if cachefile.is_file():
                g, lg = torch.load(cachefile)

        if self.transform:
            atoms = self.transform(atoms)

        if self.target == "energy_and_forces":
            target = self.get_energy_and_forces(idx)
        else:
            target = {self.target: self.df[self.target].iloc[idx]}

        target = {
            k: torch.tensor(t, dtype=torch.get_default_dtype())
            for k, t in target.items()
        }

        return atoms, target

    def get_energy_and_forces(self, idx) -> dict:

        target = {
            "energy": self.df["total_energy"][idx],
            "forces": self.df["forces"][idx],
            "stresses": self.df["stresses"][idx],
        }

        # TODO: make sure datasets use standard units...
        # data store should have total energies in eV
        if self.energy_units == "eV/atom":
            target["energy"] = target["energy"] / target["forces"].shape[0]
            # note: probably keep forces in eV/at and scale up predictions...
        elif self.energy_units == "eV":
            # data store should have total energies in eV, so do nothing
            pass

        return target

    def split_dataset_by_id(self, n_train: Union[float, int], n_val: Union[float, int]):
        """Get train/val/test split indices for SubsetRandomSampler.

        Stratify by calculation / trajectory id `"group_id"`
        """
        ids = self.df["group_id"].unique()

        # number of calculation groups
        N = len(ids)

        # test set is always the same 10 % of calculation groups
        n_test = int(0.1 * N)
        test_rng = default_rng(0)
        test_rng.shuffle(ids)
        test_ids = ids[0:n_test]
        train_val_ids = ids[n_test:-1]

        if isinstance(n_train, float) and isinstance(n_val, float):
            # n_train and n_val specified as fractions of dataset
            if n_train + n_val > 0.9:
                raise ValueError("training and validation set exceed 90% of data")
            n_val = int(n_val * N)
            n_train = int(n_train * N)

        if isinstance(n_train, int) and isinstance(n_val, int):
            # n_train and n_val specified directly as calculation group counts
            if n_train + n_val > 0.9 * N:
                raise ValueError("training and validation set exceed 90% of data")

        # configurable train/val seed
        train_val_rng = default_rng(self.train_val_seed)
        train_val_rng.shuffle(train_val_ids)

        val_ids = train_val_ids[0:n_val]
        train_ids = train_val_ids[n_val : n_val + n_train]

        # calculation ids...
        split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}

        # split atomistic configurations
        all_ids = self.df["group_id"]
        return {
            key: np.where(all_ids.isin(split_ids))[0]
            for key, split_ids in split_ids.items()
        }

    def split_dataset(self):
        """Get train/val/test split indices for SubsetRandomSampler."""
        N = len(self)
        n_test = int(0.1 * N)
        n_val = int(0.1 * N)
        # n_train = N - n_val - n_test

        # deterministic test split, always
        test_rng = default_rng(0)
        shuf = test_rng.permutation(N)
        test_ids = shuf[:n_test]
        train_val_ids = shuf[n_test:]

        # configurable train/val seed
        train_val_rng = default_rng(self.train_val_seed)
        train_val_rng.shuffle(train_val_ids)

        val_ids = train_val_ids[:n_val]
        train_ids = train_val_ids[n_val:]

        return {"train": train_ids, "val": val_ids, "test": test_ids}

    @staticmethod
    def collate_default(samples: List[Tuple[dgl.DGLGraph, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`.

        Forces get collated into a graph batch
        by concatenating along the atoms dimension

        energy and stress are global targets (properties of the whole graph)
        total energy is a scalar, stess is a rank 2 tensor


        """
        graphs, targets = map(list, zip(*samples))

        energy = torch.tensor([t["energy"] for t in targets])
        forces = torch.cat([t["forces"] for t in targets], dim=0)
        stresses = torch.stack([t["stresses"] for t in targets])

        targets = dict(total_energy=energy, forces=forces, stresses=stresses)

        batched_graph = dgl.batch(graphs)
        return batched_graph, targets

    @staticmethod
    def collate_line_graph(
        samples: List[Tuple[dgl.DGLGraph, dgl.DGLGraph, torch.Tensor]]
    ):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, line_graphs, targets = map(list, zip(*samples))

        energy = torch.tensor([t["energy"] for t in targets])
        forces = torch.cat([t["forces"] for t in targets], dim=0)
        stresses = torch.stack([t["stresses"] for t in targets])
        targets = dict(total_energy=energy, forces=forces, stresses=stresses)

        return dgl.batch(graphs), dgl.batch(line_graphs), targets
