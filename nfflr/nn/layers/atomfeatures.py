from typing import Literal
import torch

from mendeleev.fetch import fetch_table
from jarvis.core.specie import chem_data, get_node_attributes


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
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.atom_embedding = torch.nn.Embedding(108, d_model)

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

        self.rows = torch.from_numpy(table.row.values)
        self.cols = torch.from_numpy(table.col.values)

        self.row_embedding = torch.nn.Embedding(9, d_model)
        self.col_embedding = torch.nn.Embedding(18, d_model)

    def forward(self, zs):
        row = self.rows[zs - 1]
        col = self.cols[zs - 1]
        return self.row_embedding(row) + self.col_embedding(col)
