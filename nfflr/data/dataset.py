"""Standalone dataset for training force field models."""

__all__ = ()

import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Any, Literal, Union, Optional, Callable, Sequence
from typing import Protocol, runtime_checkable

import ase

# import dgl
import numpy as np
import pandas as pd
import torch
import ignite
from numpy.random import default_rng
import matplotlib.pyplot as plt

import nfflr
from nfflr.atoms import jarvis_load_atoms  # noqa:E402

# from nfflr.models.utils import EnergyScaling

from jarvis.core.atoms import Atoms as jAtoms

# make sure jarvis-tools doesn't override matplotlib backend
mpl_backend = plt.get_backend()
from jarvis.db.figshare import data as jdata  # noqa:E402

plt.switch_backend(mpl_backend)


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


def get_cachedir(scratchdir: Path | str | bool = True):
    """Get local scratch directory."""
    prefix = "nfflr-"

    if isinstance(scratchdir, bool) and scratchdir:
        scratchdir = None

    if isinstance(scratchdir, str):
        scratchdir = Path(scratchdir)

    if "SCRATCH" in os.environ:
        scratchdir = Path(os.environ.get("SCRATCH"))

    if "SLURM_JOB_ID" in os.environ:
        scratchdir = scratchdir / os.environ.get("SLURM_JOB_ID")

    if scratchdir is not None:
        os.makedirs(scratchdir, exist_ok=True)

    return tempfile.TemporaryDirectory(dir=scratchdir, prefix=prefix)


def _select_relaxation_configurations(trajectory, atol=1e-2):
    """Filter relaxation threshold by absolute energy difference threshold

    Returns pandas index values for rows to keep
    """

    order = np.argsort(trajectory.energy)
    es = trajectory.energy.values[order]
    idx = trajectory.index.values[order]

    d = np.abs(np.diff(es, prepend=es[0] - 2 * atol))

    selection = d > atol

    return idx[selection]


def collate_forcefield_targets(targets: Sequence[dict[str, torch.Tensor]]):
    """Specialized collate function for force field datasets."""
    energy = torch.tensor([t["energy"] for t in targets])
    n_atoms = torch.tensor([t["forces"].size(0) for t in targets])
    forces = torch.cat([t["forces"] for t in targets], dim=0)
    result = dict(energy=energy, n_atoms=n_atoms, forces=forces)
    if "stress" in targets[0]:
        result["stress"] = torch.stack([t["stress"] for t in targets])
    if "virial" in targets[0]:
        result["virial"] = torch.stack([t["virial"] for t in targets])

    return result


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.clone().detach()
    return torch.tensor(x, dtype=torch.get_default_dtype())


@runtime_checkable
class TensorLike(Protocol):
    """torch.Tensor protocol for auto-preprocessing with ignite"""

    def to(self, device, non_blocking):
        ...


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
        group_split_token: str = "_",
        n_train: Union[float, int] = 0.8,
        n_val: Union[float, int] = 0.1,
        energy_units: Literal["eV", "eV/atom"] = "eV",
        stress_conversion_factor: float | Literal["vasp"] | None = None,
        standardize: bool = False,
        diskcache: Optional[Path | str | bool] = None,
    ):
        """Pytorch Dataset for atomistic graphs.

        `df`: pandas dataframe from e.g. jarvis.db.figshare.data
        """

        if not isinstance(df, pd.DataFrame):
            df = _load_dataset(df)
        else:
            df = df.copy()

        example_id = df[id_tag].iloc[0]

        # by default: don't try to process record ids
        if not group_ids:
            df["group_id"] = df.loc[:, id_tag]
            df["step_id"] = np.ones(df.shape[0])
        elif "group_id" in df:
            print("keeping predefined group ids")
            pass
        else:
            # try to parse record ids to extract group and step ids
            if isinstance(example_id, int):
                df["group_id"] = df[id_tag]
                df["step_id"] = df.index
            elif "_" not in example_id:
                df["group_id"] = df[id_tag].values
                df["step_id"] = np.ones(df.shape[0])
            else:
                # split key like "JVASP-6664_main-5"
                # if there are multiple underscores - split on just the first one
                def _split_key(key):
                    pattern = f"(.+?){group_split_token}"
                    match = re.search(pattern, key)
                    return key[: match.end() - 1], key[match.end() :]

                df["group_id"], df["step_id"] = zip(*df[id_tag].apply(_split_key))

        if isinstance(df.atoms.iloc[0], dict):
            atoms = df.atoms.apply(lambda x: nfflr.Atoms(jAtoms.from_dict(x)))
        elif isinstance(df.atoms.iloc[0], nfflr.Atoms):
            atoms = df.atoms

        # store a numpy array, not a pandas series
        # if all system sizes are uniform, store as a non-ragged array
        # (so don't cast to object array)
        self.cells = torch.stack([a.cell for a in atoms])
        n_atoms = [len(at) for at in atoms]
        if np.unique(n_atoms).size > 1:
            _dtype = object
        else:
            _dtype = None
        self.positions = np.array([a.positions.numpy() for a in atoms], dtype=_dtype)
        self.numbers = np.array([a.numbers.numpy() for a in atoms], dtype=_dtype)

        self.transform = transform
        self.target = target

        self.df = df
        self.train_val_seed = train_val_seed
        self.id_tag = id_tag
        self.energy_units = energy_units
        if stress_conversion_factor == "vasp":
            # Vasp uses kBa with opposite sign convention from ASE
            # convert stress to eV/AA^3
            self.stress_conversion_factor = -0.1 * ase.units.GPa
        else:
            self.stress_conversion_factor = stress_conversion_factor

        self.ids = self.df[id_tag]

        self.split = self.split_dataset_by_id(n_train, n_val)

        if diskcache is not None and diskcache is not False:
            self.diskcache = get_cachedir(diskcache)
        else:
            self.diskcache = None

        # self.collate = self.collate_default
        self.prepare_batch = self.prepare_batch_default
        collate_target = torch.asarray
        if self.target == "energy_and_forces":
            collate_target = collate_forcefield_targets

        # either rely on users adding a dispatch for nfflr.batch
        # or implement `collate` method of transform
        collate_inputs = nfflr.batch
        if self.transform is not None:
            if hasattr(self.transform, "collate"):
                collate_inputs = self.transform.collate

        def _collate(samples: list[tuple["any", torch.Tensor]]):
            inputs, targets = map(list, zip(*samples))
            return collate_inputs(inputs), collate_target(targets)

        self.collate = _collate

        if self.target == "energy_and_forces":
            # self.collate = self.collate_forcefield
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

        # atomwise standardization
        self.standardize = standardize
        if standardize:
            self.setup_target_standardization()
            self.standardize = True

    def setup_target_standardization(self):
        self.standardize = True
        sel = self.split["train"]
        avg_n_atoms = np.mean([len(n) for n in self.numbers])
        energies = self.df[self.energy_key].iloc[sel].values
        self.scaler = EnergyScaling(energies, avg_n_atoms)

    def __len__(self):
        """Get length."""
        return self.df.shape[0]

    def __getitem__(self, idx):
        """Get AtomsDataset sample."""

        key = self.df[self.id_tag].iloc[idx]
        # atoms = self.atoms.iloc[idx]
        atoms = nfflr.Atoms(self.cells[idx], self.positions[idx], self.numbers[idx])

        n_atoms = len(atoms)
        # print(f"{key=}, {n_atoms=}")

        if self.target == "energy_and_forces":
            target = self.get_energy_and_forces(idx, n_atoms=n_atoms)
            volume = atoms.cell.det().abs().item()
            target["volume"] = volume
            target["virial"] = -volume * target["stress"]
            target = {k: to_tensor(t) for k, t in target.items()}

        else:
            target = torch.tensor(
                self.df[self.target].iloc[idx], dtype=torch.get_default_dtype()
            )

        if self.transform and self.diskcache is not None:
            cachekey = key.replace("/", "_")
            cachefile = Path(self.diskcache.name) / f"{cachekey}-{idx}.pkl"

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
            "energy": to_tensor(self.df[self.energy_key].iloc[idx]),
            "forces": to_tensor(self.df["forces"].iloc[idx]),
        }
        if self.standardize:
            target["energy"] = self.scaler.transform(target["energy"])
            target["forces"] = self.scaler.scale(target["forces"])

        if "stress" in self.df:
            target["stress"] = to_tensor(self.df["stress"].iloc[idx])
            if self.stress_conversion_factor is not None:
                target["stress"] *= self.stress_conversion_factor

            if self.standardize:
                target["stress"] = self.scaler.scale(target["stress"])

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
            e *= np.array([len(n) for n in self.numbers])

        e = e[sel]
        zs = torch.stack(
            [
                torch.bincount(torch.from_numpy(n), minlength=108)
                for n in self.numbers[sel]
            ]
        ).type(e.dtype)

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
    def prepare_batch_default(
        batch: tuple[Any, dict[str, torch.Tensor]],
        device=None,
        non_blocking=False,
    ) -> tuple[Any, dict[str, torch.Tensor]]:

        """Send batched dgl crystal graph to device."""

        def _convert_tensorlike(x: TensorLike) -> TensorLike:
            return (
                x.to(device=device, non_blocking=non_blocking)
                if device is not None
                else x
            )

        return ignite.utils.apply_to_type(batch, TensorLike, _convert_tensorlike)
