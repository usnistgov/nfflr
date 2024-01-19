"""Standalone dataset for training force field models."""

__all__ = ()

import os
import tempfile
import warnings
from pathlib import Path
from functools import partial
from typing import Any, Dict, List, Literal, Tuple, Union, Optional, Callable, Sequence

import dgl
import numpy as np
import pandas as pd
import torch
from numpy.random import default_rng
import matplotlib.pyplot as plt

from jarvis.core.atoms import Atoms as jAtoms

# make sure jarvis-tools doesn't override matplotlib backend
mpl_backend = plt.get_backend()
from jarvis.db.figshare import data as jdata  # noqa:E402

plt.switch_backend(mpl_backend)

from nfflr.data.atoms import Atoms, jarvis_load_atoms  # noqa:E402


def _load_dataset_directory(directory: Path) -> pd.DataFrame:
    """Load material dataset from directory format.

    Target properties should be listed in a csv file `id_prop.csv`
    with two columns: filename and target value.
    """
    # assume no header in file...
    # set `jid` to the file name...
    info = pd.read_csv(directory / "id_prop.csv", names=["jid", "target"])

    info["atoms"] = [jarvis_load_atoms(directory / name) for name in info["jid"]]

    return info


def _load_dataset(dataset_name, cache_dir=None):
    """Set up dataset."""

    print(f"{dataset_name=}")

    if dataset_name in (
        "alignn_ff_db",
        "dft_2d",
        "dft_3d",
        "megnet",
        "mlearn",
        "tinnet_N",
        "jff",
    ):
        df = pd.DataFrame(jdata(dataset_name, store_dir=cache_dir))
        return df

    if isinstance(dataset_name, str):
        dataset_name = Path(dataset_name)

    if dataset_name.is_file():
        # assume json or json-lines format
        # e.g., "jdft_max_min_307113_id_prop.json"
        lines = "jsonl" in dataset_name.name
        df = pd.read_json(dataset_name, lines=lines)
    elif dataset_name.is_dir():
        df = _load_dataset_directory(dataset_name)

    return df


def get_cachedir():
    """Get local scratch directory."""

    scratchdir = None
    prefix = "nfflr-"

    slurm_job = os.environ.get("SLURM_JOB_ID")
    scratchspace = os.environ.get("SCRATCH", "/scratch")
    if slurm_job is not None:
        scratchdir = f"{scratchspace}/{slurm_job}"
        if not os.path.isdir(scratchdir):
            os.makedirs(scratchdir, exist_ok=True)

    cachedir = tempfile.TemporaryDirectory(dir=scratchdir, prefix=prefix)

    return cachedir


def collate_forcefield_targets(targets: Sequence[Dict[str, torch.Tensor]]):
    """Specialized collate function for force field datasets."""
    energy = torch.tensor([t["energy"] for t in targets])
    n_atoms = torch.tensor([t["forces"].size(0) for t in targets])
    forces = torch.cat([t["forces"] for t in targets], dim=0)
    result = dict(total_energy=energy, n_atoms=n_atoms, forces=forces)
    if "stresses" in targets[0]:
        result["stresses"] = torch.stack([t["stresses"] for t in targets])
    return result


class AtomsDataset(torch.utils.data.Dataset):
    """Dataset of Atoms."""

    def __init__(
        self,
        df: Union[str, Path, pd.DataFrame],
        target: str = "formation_energy_peratom",
        transform: Optional[Callable] = None,
        custom_collate_fn: Optional[Callable] = None,
        custom_prepare_batch_fn: Optional[Callable] = None,
        train_val_seed: int = 42,
        id_tag: str = "jid",
        group_ids: bool = False,
        n_train: Union[float, int] = 0.8,
        n_val: Union[float, int] = 0.1,
        energy_units: Literal["eV", "eV/atom"] = "eV/atom",
        diskcache: bool = False,
    ):
        """Pytorch Dataset for atomistic graphs.

        `df`: pandas dataframe from e.g. jarvis.db.figshare.data
        """

        if not isinstance(df, pd.DataFrame):
            df = _load_dataset(df)

        example_id = df[id_tag].iloc[0]

        # by default: don't try to process record ids
        if not group_ids:
            df["group_id"] = df[id_tag]
            df["step_id"] = np.ones(df.shape[0])
        else:
            # try to parse record ids to extract group and step ids
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

        if isinstance(df.atoms.iloc[0], dict):
            self.atoms = df.atoms.apply(lambda x: Atoms(jAtoms.from_dict(x)))
        elif isinstance(df.atoms.iloc[0], Atoms):
            self.atoms = df.atoms

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

        self.collate = self.collate_default
        self.prepare_batch = self.prepare_batch_default

        if self.target == "energy_and_forces":
            self.collate = self.collate_forcefield
            if "total_energy" in self.df.keys():
                self.energy_key = "total_energy"
            else:
                self.energy_key = "energy"
        else:
            self.energy_key = self.target

        if callable(custom_collate_fn):
            self.collate = custom_collate_fn

        if callable(custom_prepare_batch_fn):
            self.prepare_batch = custom_prepare_batch_fn

    def __len__(self):
        """Get length."""
        return self.df.shape[0]

    def __getitem__(self, idx):
        """Get AtomsDataset sample."""

        key = self.df[self.id_tag].iloc[idx]
        atoms = self.atoms.iloc[idx]
        n_atoms = len(atoms)
        # print(f"{key=}, {n_atoms=}")

        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.clone().detach()
            return torch.tensor(x, dtype=torch.get_default_dtype())

        if self.target == "energy_and_forces":
            target = self.get_energy_and_forces(idx, n_atoms=n_atoms)
            # volume: abs(determinant(cell))
            target["volume"] = atoms.lattice.det().abs().item()
            target = {k: to_tensor(t) for k, t in target.items()}

        else:
            target = torch.tensor(
                self.df[self.target].iloc[idx], dtype=torch.get_default_dtype()
            )

        if self.transform and self.diskcache is not None:
            cachefile = Path(self.diskcache.name) / f"jarvis-{key}-{idx}.pkl"

            if cachefile.is_file():
                # print(f"loading from {cachefile}")
                atoms = torch.load(cachefile)
            else:
                # print(f"saving to {cachefile}")
                atoms = self.transform(atoms)
                torch.save(atoms, cachefile)
        elif self.transform:
            atoms = self.transform(atoms)

        return atoms, target

    def get_energy_and_forces(self, idx, n_atoms) -> dict:
        target = {
            "energy": self.df[self.energy_key].iloc[idx],
            "forces": self.df["forces"].iloc[idx],
        }
        if "stresses" in self.df:
            target["stresses"] = self.df["stresses"].iloc[idx]

        # TODO: make sure datasets use standard units...
        # data store should have total energies in eV
        if self.energy_units == "eV/atom":
            target["energy"] = target["energy"] * len(target["forces"])
            # note: probably keep forces in eV/at and scale up predictions...
        elif self.energy_units == "eV":
            # data store should have total energies in eV, so do nothing
            pass

        return target

    def estimate_reference_energies(self, use_bias=False):

        sel = self.split["train"]

        e = torch.from_numpy(self.df[self.energy_key].values)
        if self.energy_units == "eV/atom":
            e *= self.atoms.apply(len).values

        e = e[sel]
        zs = self.atoms[sel].apply(lambda a: torch.bincount(a.numbers, minlength=108))
        zs = torch.stack(zs.values.tolist()).type(e.dtype)

        if use_bias:
            # add constant for zero offset
            zs[:, 0] = 1

        return torch.linalg.lstsq(zs, e).solution

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
                warnings.warn("training and validation set exceed 90% of data")
            n_val = int(n_val * N)
            n_train = int(n_train * N)

        if isinstance(n_train, int) and isinstance(n_val, int):
            # n_train and n_val specified directly as calculation group counts
            if n_train + n_val > 0.9 * N:
                warnings.warn("training and validation set exceed 90% of data")

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
    def collate_forcefield(samples: List[Tuple[dgl.DGLGraph, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`.

        Forces get collated into a graph batch
        by concatenating along the atoms dimension

        energy and stress are global targets (properties of the whole graph)
        total energy is a scalar, stess is a rank 2 tensor


        """
        graphs, targets = map(list, zip(*samples))

        targets = collate_forcefield_targets(targets)

        batched_graph = dgl.batch(graphs)
        return batched_graph, targets

    @staticmethod
    def prepare_batch_default(
        batch: Tuple[Any, Dict[str, torch.Tensor]],
        device=None,
        non_blocking=False,
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """Send batched dgl crystal graph to device."""
        atoms, targets = batch
        if isinstance(targets, torch.Tensor):
            targets = targets.to(device, non_blocking=non_blocking)
        else:
            targets = {
                k: v.to(device, non_blocking=non_blocking) for k, v in targets.items()
            }

        if isinstance(atoms, list):
            g, lg = atoms
            batch = ((g.to(device, non_blocking=non_blocking), lg.to(device, non_blocking=non_blocking)), targets)
        else:
            batch = (atoms.to(device, non_blocking=non_blocking), targets)

        return batch

    @staticmethod
    def collate_default(samples: List[Tuple[dgl.DGLGraph, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`.

        Forces get collated into a graph batch
        by concatenating along the atoms dimension

        energy and stress are global targets (properties of the whole graph)
        total energy is a scalar, stess is a rank 2 tensor


        """
        graphs, targets = map(list, zip(*samples))
        return dgl.batch(graphs), torch.tensor(targets)

    @staticmethod
    def collate_default_line_graph(samples: List[Tuple[dgl.DGLGraph, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`.

        Forces get collated into a graph batch
        by concatenating along the atoms dimension

        energy and stress are global targets (properties of the whole graph)
        total energy is a scalar, stess is a rank 2 tensor


        """
        inputs, targets = map(list, zip(*samples))
        graphs, line_graphs = map(list, zip(*inputs))
        return (dgl.batch(graphs), dgl.batch(line_graphs)), torch.tensor(targets)


    @staticmethod
    def collate_line_graph_ff(
        samples: List[Tuple[dgl.DGLGraph, dgl.DGLGraph, torch.Tensor]]
    ):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, line_graphs, targets = map(list, zip(*samples))

        energy = torch.tensor([t["energy"] for t in targets])
        forces = torch.cat([t["forces"] for t in targets], dim=0)
        stresses = torch.stack([t["stresses"] for t in targets])
        targets = dict(total_energy=energy, forces=forces, stresses=stresses)

        return dgl.batch(graphs), dgl.batch(line_graphs), targets
