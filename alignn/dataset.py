"""Standalone dataset for training force field models."""
import os
import pickle
import shutil
from functools import partial
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

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
    all_ranges = [
        torch.arange(x, y, dtype=precision) for x, y in zip(nmin, nmax)
    ]
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

    if torch.get_default_dtype() == torch.float32:
        try:
            g.ndata["coord"] = X_src.float()
            g.edata["r"] = (X_dst[v] - X_src[src]).float()
        except:
            print(a)
            print(g)
            print(X_src)
            raise
    else:
        g.ndata["coord"] = X_src
        g.edata["r"] = X_dst[v] - X_src[src]

    g.ndata["atom_features"] = torch.tensor(a.atomic_numbers)[:, None]

    return g


def atom_dgl_multigraph_torch(
    atoms: Atoms,
    cutoff: float = 8,
    bond_tol: float = 0.15,
    atol=1e-5,
    topk_tol=1.001,
    precision=torch.float64,
):
    """
    Get neighbors for each atom in the unit cell, out to a distance r.

    Contains [index_i, index_j, distance, image] array.

    This function builds a supercell and identifies all edges  by
    brute force calculation of the pairwise distances between atoms
    in the original cell and atoms in the supercell. If the kNN graph
    construction is used, do some extra work to make sure that all
    edges have a reverse pair for dgl undirected graph representation,
    but also that edges are not double counted.

    # check that torch knn graph is equivalent to pytorch version (with canonical edges)
    a = Atoms(...)
    pg = graphs.Graph.atom_dgl_multigraph(a, use_canonize=True, compute_line_graph=False)
    tg = graphs.Graph.atom_dgl_multigraph_torch(a, compute_line_graph=False)

    # round bond displacement vectors to add tolerance to numerical error
    pg_edata = list(
        zip(
            map(int, pg.edges()[0]),
            map(int, pg.edges()[1]),
            map(tuple, torch.round(pg.edata["r"], decimals=3).tolist()),
        )
    )
    tg_edata = list(
        zip(
            map(int, tg.edges()[0]),
            map(int, tg.edges()[1]),
            map(tuple, torch.round(tg.edata["r"], decimals=3).tolist()),
        )
    )
    set(tg_edata).difference(pg_edata) # -> yields empty set

    """

    if atoms is not None:
        cart_coords = torch.tensor(atoms.cart_coords, dtype=precision)
        frac_coords = torch.tensor(atoms.frac_coords, dtype=precision)
        lattice_mat = torch.tensor(atoms.lattice_mat, dtype=precision)
        elements = atoms.elements

    X_src = cart_coords
    num_atoms = X_src.shape[0]

    # determine how many supercells are needed for the cutoff radius
    recp = 2 * torch.pi * torch.linalg.inv(lattice_mat).T
    recp_len = torch.tensor(
        [i for i in (torch.sqrt(torch.sum(recp**2, dim=1)))]
    )

    maxr = torch.ceil((cutoff + bond_tol) * recp_len / (2 * torch.pi))
    nmin = torch.floor(torch.min(frac_coords, dim=0)[0]) - maxr
    nmax = torch.ceil(torch.max(frac_coords, dim=0)[0]) + maxr

    # construct the supercell index list
    all_ranges = [
        torch.arange(x, y, dtype=precision) for x, y in zip(nmin, nmax)
    ]
    cell_images = torch.cartesian_prod(*all_ranges)

    # tile periodic images into X_dst
    # index id_dst into X_dst maps to atom id as id_dest % num_atoms
    X_dst = (cell_images @ lattice_mat)[:, None, :] + X_src
    X_dst = X_dst.reshape(-1, 3)

    # pairwise distances between atoms in (0,0,0) cell
    # and atoms in all periodic images
    dist = torch.cdist(X_src, X_dst)

    # radius graph
    neighbor_mask = torch.bitwise_and(
        dist <= cutoff,
        ~torch.isclose(dist, torch.DoubleTensor([0]), atol=atol),
    )
    # get node indices for edgelist from neighbor mask
    src, v = torch.where(neighbor_mask)

    # index into tiled cell image index to atom ids
    g = dgl.graph((src, v % num_atoms))
    g.ndata["cart_coords"] = X_src.float()
    g.ndata["frac_coords"] = frac_coords.float()
    g.gdata = lattice_mat
    g.edata["r"] = (X_dst[v] - X_src[src]).float()
    g.edata["X_src"] = X_src[src]
    g.edata["X_dst"] = X_dst[v]
    g.edata["src"] = src

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

    build (and cache) graphs on each access instead of precomputing them
    also, make sure to split sections grouped on id column

    example_data = Path("alignn/examples/sample_data")
    df = pd.read_json(example_data / "id_prop.json")

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
        cutoff_radius: float = 6.0,
        n_train: Union[float, int] = 0.8,
        n_val: Union[float, int] = 0.1,
        neighbor_strategy: Literal["cutoff", "12nn"] = "cutoff",
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
        self.cutoff_radius = cutoff_radius
        self.neighbor_strategy = neighbor_strategy
        self.energy_units = energy_units

        if self.line_graph:
            self.collate = self.collate_line_graph
            self.prepare_batch = prepare_line_graph_batch
        else:
            self.collate = self.collate_default
            self.prepare_batch = prepare_dgl_batch

        self.ids = self.df[id_tag]

        self.split = self.split_dataset_by_id(n_train, n_val)

        # features = self._get_attribute_lookup(atom_features)
        self.lmdb_name = "jv_300k_scratch.db"
        # self.lmdb_name = "jv_2M.db"
        self.lmdb_path = Path("data")

        scratch = get_scratch_dir()
        scratch.mkdir(exist_ok=True)
        if (self.lmdb_path / self.lmdb_name).exists():
            shutil.copytree(
                self.lmdb_path / self.lmdb_name, scratch / self.lmdb_name
            )
        self.lmdb_scratch_path = str(scratch / self.lmdb_name)

        self.lmdb_sz = int(1e11)
        self.env = None

    def load_graph(self, idx: int):
        """Deserialize graph from lmdb store using calculation key."""
        key = self.df[self.id_tag].iloc[idx]

        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_scratch_path), map_size=self.lmdb_sz
            )

        with self.env.begin() as txn:
            record = txn.get(key.encode())

        if record is not None:
            g, lg = pickle.loads(record)

        else:
            a = Atoms.from_dict(self.df["atoms"].iloc[idx])
            g = build_radius_graph_torch(
                a,
                neighbor_strategy=self.neighbor_strategy,
                r=self.cutoff_radius,
            )

            if self.line_graph:
                lg = g.line_graph(shared=False)
            else:
                lg = None

            with self.env.begin(write=True) as txn:
                txn.put(key.encode(), pickle.dumps((g, lg)))

        # don't serialize redundant edge data...
        if self.line_graph:
            lg.ndata["r"] = g.edata["r"]

        return g, lg

    def split_dataset_by_id(
        self, n_train: Union[float, int], n_val: Union[float, int]
    ):
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
                raise ValueError(
                    "training and validation set exceed 90% of data"
                )
            n_val = int(n_val * N)
            n_train = int(n_train * N)

        if isinstance(n_train, int) and isinstance(n_val, int):
            # n_train and n_val specified directly as calculation group counts
            if n_train + n_val > 0.9 * N:
                raise ValueError(
                    "training and validation set exceed 90% of data"
                )

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

        # JVASP-119960_main-1 causing problems
        # also JVASP-118572_main-1
        # this is because the shorted bond is greater than the cutoff length
        # key = self.df[self.id_tag].iloc[idx]
        # print(key)

        g, lg = self.load_graph(idx)

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

        if self.line_graph:
            # lg = g.line_graph(shared=True)
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
