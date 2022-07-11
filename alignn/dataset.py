"""Standalone dataset for training force field models."""
import os
import pickle
import shutil
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
from tqdm import tqdm

from alignn.graphs import Graph


def get_scratch_dir():
    """Get local scratch directory."""
    scratch = Path("/tmp/alignnff")

    slurm_job = os.environ.get("SLURM_JOB_ID")
    if slurm_job is not None:
        scratch = Path(f"/scratch/{slurm_job}")

    return scratch


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


def build_radius_graph_torch(
    a: Atoms,
    r: float = 5,
    bond_tol: float = 0.15,
    neighbor_strategy: Literal["cutoff", "12nn"] = "cutoff",
):
    """
    Get neighbors for each atom in the unit cell, out to a distance r.
    Contains [index_i, index_j, distance, image] array.
    Adapted from jarvis-tools, in turn adapted from pymatgen.

    Optionally, use a 12th-neighbor-shell graph

    might be differentiable wrt atom coords, definitely not wrt cell params
    """
    precision = torch.float64
    atol = 1e-5

    n = a.num_atoms
    X_src = torch.tensor(a.cart_coords, dtype=precision)
    lattice_matrix = torch.tensor(a.lattice_mat, dtype=precision)

    # cutoff -> calculate which periodic images to consider
    recp_len = np.array(a.lattice.reciprocal_lattice().abc)
    maxr = np.ceil((r + bond_tol) * recp_len / (2 * np.pi))
    nmin = np.floor(np.min(a.frac_coords, axis=0)) - maxr
    nmax = np.ceil(np.max(a.frac_coords, axis=0)) + maxr
    all_ranges = [torch.arange(x, y, dtype=precision) for x, y in zip(nmin, nmax)]
    cell_images = torch.cartesian_prod(*all_ranges)

    # tile periodic images into X_dst
    # index id_dst into X_dst maps to atom id as id_dest % num_atoms
    X_dst = (cell_images @ lattice_matrix)[:, None, :] + X_src
    X_dst = X_dst.reshape(-1, 3)

    # pairwise distances between atoms in (0,0,0) cell
    # and atoms in all periodic images
    dist = torch.cdist(X_src, X_dst)

    if neighbor_strategy == "cutoff":
        neighbor_mask = torch.bitwise_and(
            dist <= r, ~torch.isclose(dist, torch.DoubleTensor([0]), atol=atol)
        )

    elif neighbor_strategy == "12nn":
        # collect 12th-nearest neighbor distance
        # topk: k = 13 because first neighbor is a self-interaction
        # this is filtered out in the neighbor_mask selection
        nbrdist, _ = dist.topk(13, largest=False)
        k_dist = nbrdist[:, -1]

        # expand k-NN graph to include all atoms in the
        # neighbor shell of the twelfth neighbor
        # broadcast the <= along the src axis
        neighbor_mask = torch.bitwise_and(
            dist <= 1.05 * k_dist[:, None],
            ~torch.isclose(dist, torch.DoubleTensor([0]), atol=atol),
        )

    # get node indices for edgelist from neighbor mask
    src, v = torch.where(neighbor_mask)

    # index into tiled cell image index to atom ids
    g = dgl.graph((src, v % n))
    g.ndata["coord"] = X_src.float()
    g.edata["r"] = (X_dst[v] - X_src[src]).float()
    # print(torch.norm(g.edata["r"], dim=1).sort()[0])
    g.ndata["atom_features"] = torch.tensor(a.atomic_numbers)[:, None]

    return g


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
        energy_units: Literal["eV", "eV/atom"] = "eV/atom",
    ):
        """Pytorch Dataset for atomistic graphs.

        `df`: pandas dataframe from e.g. jarvis.db.figshare.data
        """
        example_id = df[id_tag].iloc[0]

        if isinstance(example_id, int):
            df["group_id"] = df[id_tag]
            df["step_id"] = df.index
        else:
            # split key like "JVASP-6664_main-5"
            df["group_id"], df["step_id"] = zip(
                *df[id_tag].apply(partial(str.split, sep="_"))
            )

        self.df = df
        self.line_graph = line_graph
        self.train_val_seed = train_val_seed
        self.id_tag = id_tag
        self.energy_units = energy_units

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
        self.lmdb_name = "jv_300k_scratch.db"
        # self.lmdb_name = "jv_2M.db"
        self.lmdb_path = Path("data")

        scratch = get_scratch_dir()
        scratch.mkdir(exist_ok=True)
        if (self.lmdb_path / self.lmdb_name).exists():
            shutil.copytree(self.lmdb_path / self.lmdb_name, scratch / self.lmdb_name)
        self.lmdb_scratch_path = str(scratch / self.lmdb_name)

        self.lmdb_sz = int(1e10)
        self.env = None
        # self.produce_graphs()

    def load_graph(self, idx: int):
        """Deserialize graph from lmdb store using calculation key."""
        key = self.df[self.id_tag].iloc[idx]
        # print(idx, key)

        if self.env is None:
            self.env = lmdb.open(str(self.lmdb_scratch_path), map_size=self.lmdb_sz)

        with self.env.begin() as txn:
            record = txn.get(key.encode())
        if record is not None:
            g = pickle.loads(record)
        else:
            # g = atoms_to_graph(self.df["atoms"].iloc[idx])
            a = Atoms.from_dict(self.df["atoms"].iloc[idx])
            # print(key)
            g = build_radius_graph_torch(a, neighbor_strategy="12nn")
            with self.env.begin(write=True) as txn:
                txn.put(key.encode(), pickle.dumps(g))
        return g

    def produce_graphs(self):
        """Precompute graphs. store pickled graphs in lmdb store."""
        print("precomputing atomistic graphs")
        env = lmdb.open(self.lmdb_scratch_path, map_size=self.lmdb_sz)

        # skip anything already cached
        with env.begin() as txn:
            cached = set(map(bytes.decode, txn.cursor().iternext(values=False)))

        to_compute = set(self.ids).difference(cached)
        uncached = self.df[self.ids.isin(to_compute)]

        if len(uncached) == 0:
            return
        else:
            print(f"precomputing {len(uncached)}/{len(self.ids)} graphs")

        cols = (self.id_tag, "atoms")
        for idx, jid, atoms in tqdm(
            uncached.loc[:, cols].itertuples(name=None), total=len(uncached)
        ):
            graph = atoms_to_graph(atoms)
            with env.begin(write=True) as txn:
                txn.put(jid.encode(), pickle.dumps(graph))

        # if there are any updates to the scratch lmdb store, sync back
        shutil.copytree(
            self.lmdb_scratch_path,
            self.lmdb_path / self.lmdb_name,
            dirs_exist_ok=True,
        )

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
        # key = self.df[self.id_tag].iloc[idx]
        # g = self.load_graph(key)
        g = self.load_graph(idx)

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

        # TODO: make sure datasets use standard units...
        # data store should have total energies in eV
        if self.energy_units == "eV/atom":
            target["energy"] = target["energy"] / g.num_nodes()
            # note: probably keep forces in eV/at and scale up predictions...
            # target["forces"] = target["forces"] / g.num_nodes()
        elif self.energy_units == "eV":
            # data store should have total energies in eV, so do nothing
            pass

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
