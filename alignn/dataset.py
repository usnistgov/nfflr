"""Standalone dataset for training force field models."""
import os
import shutil
import pickle
from tqdm import tqdm
from functools import partial
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import dgl
import lmdb
import numpy as np
import pandas as pd
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import (
    chem_data,
    compute_bond_cosines,
    get_node_attributes,
)
from numpy.random import default_rng

from alignn.graphs import Graph


def atoms_to_graph(atoms):
    """Convert structure dict to DGLGraph."""
    structure = Atoms.from_dict(atoms)
    return Graph.atom_dgl_multigraph(
        structure,
        cutoff=8.0,
        atom_features="atomic_number",
        max_neighbors=12,
        compute_line_graph=False,
        use_canonize=True,
    )


def prepare_line_graph_batch(
    batch: Tuple[dgl.DGLGraph, dgl.DGLGraph, Dict[str, torch.Tensor]],
    device=None,
    non_blocking=False,
) -> Tuple[Tuple[dgl.DGLGraph, dgl.DGLGraph], Dict[str, torch.Tensor]]:
    """Send batched dgl crystal graph to device."""
    g, lg, t = batch
    t = {k: v.to(device, non_blocking=non_blocking) for k, v in t.items()}

    batch = (
        (
            g.to(device, non_blocking=non_blocking),
            lg.to(device, non_blocking=non_blocking),
        ),
        t,
    )

    return batch


def prepare_dgl_batch(
    batch: Tuple[dgl.DGLGraph, Dict[str, torch.Tensor]],
    device=None,
    non_blocking=False,
) -> Tuple[dgl.DGLGraph, Dict[str, torch.Tensor]]:
    """Send batched dgl crystal graph to device."""
    g, t = batch
    t = {k: v.to(device, non_blocking=non_blocking) for k, v in t.items()}

    batch = (g.to(device, non_blocking=non_blocking), t)

    return batch


class AtomisticConfigurationDataset(torch.utils.data.Dataset):
    """Dataset of crystal DGLGraphs.

    target: total_energy, forces, stresses
    """

    def __init__(
        self,
        df: pd.DataFrame,
        atom_features: str = "atomic_number",
        transform: bool = None,
        line_graph: bool = False,
        train_val_seed: int = 42,
        id_tag: str = "jid",
    ):
        """Pytorch Dataset for atomistic graphs.

        `df`: pandas dataframe from e.g. jarvis.db.figshare.data
        """
        # split key like "JVASP-6664_main-5"
        df["group_id"], df["step_id"] = zip(
            *df[id_tag].apply(partial(str.split, sep="_"))
        )

        self.df = df
        self.line_graph = line_graph
        self.train_val_seed = train_val_seed
        self.id_tag = id_tag

        if self.line_graph:
            self.collate = self.collate_line_graph
            self.prepare_batch = prepare_line_graph_batch
        else:
            self.collate = self.collate_default
            self.prepare_batch = prepare_dgl_batch

        self.ids = self.df[id_tag]
        self.transform = transform

        self.split = self.split_dataset_by_id()

        # features = self._get_attribute_lookup(atom_features)
        self.lmdb_name = "jv_300k.db"
        self.lmdb_path = Path("data")

        scratch = Path(f"/scratch/{os.environ.get('SLURM_JOB_ID')}")
        shutil.copytree(self.lmdb_path / self.lmdb_name, scratch / self.lmdb_name)
        self.lmdb_scratch_path = str(scratch / self.lmdb_name)

        self.lmdb_sz = int(1e10)
        self.env = None
        self.produce_graphs()

    def load_graph(self, key: str):
        """Deserialize graph from lmdb store using calculation key."""
        if self.env is None:
            self.env = lmdb.open(str(self.lmdb_scratch_path), map_size=self.lmdb_sz)

        with self.env.begin() as txn:
            g = pickle.loads(txn.get(key.encode()))
        return g

    def produce_graphs(self):
        """Precompute graphs. store pickled graphs in lmdb store."""
        print("precomputing atomistic graphs")
        env = lmdb.open(self.lmdb_scratch_path, map_size=self.lmdb_sz)
        # env = lmdb.open(str(self.lmdb_path / self.lmdb_name), map_size=self.lmdb_sz)

        # skip anything already cached
        with env.begin() as txn:
            cached = set(map(bytes.decode, txn.cursor().iternext(values=False)))

        to_compute = set(self.ids).difference(cached)
        uncached = self.df[self.ids.isin(to_compute)]

        cols = (self.id_tag, "atoms")
        for idx, jid, atoms in tqdm(
            uncached.loc[:, cols].itertuples(name=None), total=len(uncached)
        ):
            graph = atoms_to_graph(atoms)
            with env.begin(write=True) as txn:
                txn.put(jid.encode(), pickle.dumps(graph))

        # shutil.copytree(
        #     self.lmdb_scratch_path, self.lmdb_path / self.lmdb_name, dirs_exist_ok=True
        # )

        # graphs = self.df["atoms"].apply(atoms_to_graph).values
        # self.graphs = graphs

        # if self.line_graph:
        #     self.prepare_batch = prepare_line_graph_batch

        #     print("building line graphs")
        #     self.line_graphs = []
        #     for g in self.graphs:
        #         lg = g.line_graph(shared=True)
        #         lg.apply_edges(compute_bond_cosines)
        #         self.line_graphs.append(lg)

        # # load selected node representation
        # # assume graphs contain atomic number in g.ndata["atom_features"]
        # for g in graphs:
        #     z = g.ndata.pop("atom_features")
        #     g.ndata["atomic_number"] = z
        #     z = z.type(torch.IntTensor).squeeze()
        #     f = torch.tensor(features[z]).type(torch.FloatTensor)
        #     if g.num_nodes() == 1:
        #         f = f.unsqueeze(0)
        #     g.ndata["atom_features"] = f

    def split_dataset_by_id(self):
        """Get train/val/test split indices for SubsetRandomSampler.

        Stratify by calculation / trajectory id `"group_id"`
        """
        ids = self.df["group_id"].unique()

        # number of calculation groups
        N = len(ids)
        n_test = int(0.1 * N)
        n_val = int(0.1 * N)
        # n_train = N - n_val - n_test

        # deterministic test split, always
        test_rng = default_rng(0)
        test_rng.shuffle(ids)
        test_ids = ids[:n_test]
        train_val_ids = ids[n_test:]

        # configurable train/val seed
        train_val_rng = default_rng(self.train_val_seed)
        train_val_rng.shuffle(train_val_ids)

        val_ids = train_val_ids[:n_val]
        train_ids = train_val_ids[n_val:]

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
    def _get_attribute_lookup(atom_features: str = "cgcnn"):
        """Build a lookup array indexed by atomic number."""
        max_z = max(v["Z"] for v in chem_data.values())

        # get feature shape (referencing Carbon)
        template = get_node_attributes("C", atom_features)

        features = np.zeros((1 + max_z, len(template)))

        for element, v in chem_data.items():
            z = v["Z"]
            x = get_node_attributes(element, atom_features)

            if x is not None:
                features[z, :] = x

        return features

    def __len__(self):
        """Get length."""
        return self.df.shape[0]

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        # g = self.graphs[idx]
        # g = atoms_to_graph(self.df["atoms"].iloc[idx])
        key = self.df[self.id_tag].iloc[idx]
        g = self.load_graph(key)

        g.ndata["atomic_number"] = g.ndata["atom_features"]

        target = {
            "energy": self.df["total_energy"][idx],
            "forces": self.df["forces"][idx],
            "stresses": self.df["stresses"][idx],
        }

        target = {
            k: torch.tensor(t, dtype=torch.get_default_dtype())
            for k, t in target.items()
        }

        if self.transform:
            g = self.transform(g)

        if self.line_graph:
            # lg = self.line_graphs[idx]
            lg = g.line_graph(shared=True)
            # lg.apply_edges(compute_bond_cosines)
            return g, lg, target

        return g, target

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
