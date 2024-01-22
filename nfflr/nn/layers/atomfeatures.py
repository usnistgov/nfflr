from typing import Literal
import torch
import numpy as np

import einops

from mendeleev.fetch import fetch_table
from jarvis.core.specie import chem_data, get_node_attributes


class AtomType(torch.nn.Module):
    """Compact atom type lookup table."""

    def __init__(self, species: torch.Tensor, one_hot: bool = False):
        """Compact atom type lookup table.

        species should be an int tensor of unique atomic numbers
        """
        super().__init__()

        self.one_hot = one_hot

        # map atomic numbers to species ids
        self.ids = {z.item(): idx for idx, z in enumerate(species)}

        # tensorized version of species id map - store in zero-indexed format
        # initialize to -1 to raise errors on out-of-domain species
        _weight = -1 * torch.ones(118, dtype=int)
        for z, id in self.ids.items():
            _weight[z - 1] = id

        self.species = torch.nn.Embedding(
            118, 1, _weight=_weight.unsqueeze(1), _freeze=True
        )

    def __len__(self):
        return len(self.ids)

    def forward(self, z: torch.Tensor):

        # z-1: data for atomic species stored in zero-indexed format
        idx = self.species(z - 1).squeeze()

        if self.one_hot:
            return torch.nn.functional.one_hot(idx, num_classes=len(self))

        return idx.squeeze()


class AtomPairType(torch.nn.Module):
    """Compact atom pair type lookup table."""

    def __init__(
        self, species: torch.Tensor, symmetric: bool = True, one_hot: bool = False
    ):
        """Compact atom pair type lookup table.

        species should be an int tensor of unique atomic numbers
        """
        super().__init__()
        self.symmetric = symmetric
        self.one_hot = one_hot

        self.atomtypes = AtomType(species, one_hot=False)
        n = len(self.atomtypes)

        self.num_classes = n**2

        if symmetric:
            self.num_classes = n * (n + 1) // 2
            # just store a symmetric lookup table for the linear index
            # into the triangular matrix of atom type pairs
            # instead of sorting atom ids at runtime, just multi-index
            a = torch.zeros((n, n), dtype=int)
            for idx, (r, c) in enumerate(zip(*np.triu_indices(n))):
                a[r, c] = idx

            self.register_buffer("ids", a + torch.triu(a, diagonal=1).T)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):

        n = len(self.atomtypes)

        # concatenate atomic numbers into pairs
        zs, ps = einops.pack((z1, z2), "i *")

        # look up atom type ids for pairs
        atomtypes = self.atomtypes(zs)

        # calculate 2D pair index
        ia, ib = einops.unpack(atomtypes, ps, "i *")
        # pairtype = torch.from_numpy(np.ravel_multi_index((ia, ib), (n, n)))

        # convert to pair index
        if self.symmetric:
            pairtype = self.ids[ia, ib]
        else:
            pairtype = ia * n + ib

        if self.one_hot:
            return torch.nn.functional.one_hot(pairtype, num_classes=self.num_classes)

        return pairtype


def _get_attribute_lookup(atom_features: str = "cgcnn"):
    """Build a lookup array indexed by atomic number.

    atom_features can be
    cgcnn: standard one-hot encoded CGCNN atom embeddings
    basic: similar feature set, real values instead
    cfid: 438 chemical features
    """
    max_z = max(v["Z"] for v in chem_data.values())

    # get feature shape (referencing Carbon)
    template = get_node_attributes("C", atom_features)
    features = torch.zeros((1 + max_z, len(template)))

    for element, v in chem_data.items():
        z = v["Z"]
        x = get_node_attributes(element, atom_features)

        if x is not None:
            features[z, :] = torch.tensor(x)

    # replace missing features with NaN
    features[(features == -9999).any(dim=1)] = torch.nan

    # create fixed feature embedding indexed by atomic number
    n_elements, embedding_dim = features.shape
    features = torch.nn.Embedding(
        num_embeddings=n_elements,
        embedding_dim=embedding_dim,
        _weight=features,
    )
    features.weight.requires_grad = False

    return features


class AttributeEmbedding(torch.nn.Module):
    def __init__(
        self,
        atom_features: Literal["cgcnn", "basic", "cfid"] = "cgcnn",
        d_model: int = 64,
    ):
        super().__init__()

        f = _get_attribute_lookup(atom_features=atom_features)
        self.atom_embedding = torch.nn.Sequential(
            f, torch.nn.Linear(f.embedding_dim, d_model)
        )

    def forward(self, x: torch.Tensor):
        return self.atom_embedding(x)


class AtomicNumberEmbedding(torch.nn.Module):
    def __init__(self, d_model: int = 64, num_embeddings: int = 118):
        super().__init__()
        self.atom_embedding = torch.nn.Embedding(num_embeddings, d_model)

    def forward(self, x):
        return self.atom_embedding(x)


class PeriodicTableEmbedding(torch.nn.Module):
    """Additive row and column embedding."""

    def __init__(self, d_model: int = 64):
        super().__init__()

        table = fetch_table("elements")

        # build static lookup table mapping Z -> row, column
        # follow CGCNN and treat lanthanides and actinides as rows 8, 9

        f_mask = table.block == "f"
        row = table.period.copy()
        row[f_mask] += 2
        table["row"] = row

        col = table.group_id.copy()
        col[f_mask] = (
            table[f_mask]
            .groupby("period")
            .apply(lambda x: x["atomic_number"] - x["atomic_number"].min())
            .to_numpy()
            + 3
        )
        table["col"] = col.astype(int)

        self.register_buffer("rows", torch.from_numpy(table.row.values))
        self.register_buffer("cols", torch.from_numpy(table.col.values))

        self.row_embedding = torch.nn.Embedding(9, d_model)
        self.col_embedding = torch.nn.Embedding(18, d_model)

    def forward(self, zs):
        row = self.rows[zs - 1]
        col = self.cols[zs - 1]
        return self.row_embedding(row) + self.col_embedding(col)
