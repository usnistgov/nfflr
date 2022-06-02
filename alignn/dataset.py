"""Standalone dataset for training force field models."""
from typing import Dict, List, Optional, Sequence, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import (
    chem_data,
    compute_bond_cosines,
    get_node_attributes,
)

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
        graphs: Optional[Sequence[dgl.DGLGraph]] = None,
        atom_features="atomic_number",
        transform=None,
        line_graph=False,
        classification=False,
        id_tag="jid",
    ):
        """Pytorch Dataset for atomistic graphs.

        `df`: pandas dataframe from e.g. jarvis.db.figshare.data
        `graphs`: DGLGraph representations corresponding to rows in `df`
        `target`: key for label column in `df`
        """
        self.df = df
        self.graphs = graphs
        self.line_graph = line_graph
        if self.line_graph:
            self.collate = self.collate_line_graph
        else:
            self.collate = self.collate_default

        self.ids = self.df[id_tag]
        self.transform = transform

        features = self._get_attribute_lookup(atom_features)

        if graphs is None:
            graphs = df["atoms"].apply(atoms_to_graph).values

        # load selected node representation
        # assume graphs contain atomic number in g.ndata["atom_features"]
        for g in graphs:
            z = g.ndata.pop("atom_features")
            g.ndata["atomic_number"] = z
            z = z.type(torch.IntTensor).squeeze()
            f = torch.tensor(features[z]).type(torch.FloatTensor)
            if g.num_nodes() == 1:
                f = f.unsqueeze(0)
            g.ndata["atom_features"] = f

        self.graphs = graphs

        self.prepare_batch = prepare_dgl_batch
        if line_graph:
            self.prepare_batch = prepare_line_graph_batch

            print("building line graphs")
            self.line_graphs = []
            for g in self.graphs:
                lg = g.line_graph(shared=True)
                lg.apply_edges(compute_bond_cosines)
                self.line_graphs.append(lg)

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
        g = self.graphs[idx]

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
            return g, self.line_graphs[idx], target

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
