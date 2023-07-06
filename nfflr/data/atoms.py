from pathlib import Path
from collections.abc import Iterable

import ase
import dgl
import torch
from torch import nn
from typing import TypeAlias, Optional, List
import jarvis.core.atoms
from jarvis.core.specie import chem_data, get_node_attributes

from plum import dispatch

Z_dtype = torch.int


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
    features = nn.Embedding(
        num_embeddings=n_elements,
        embedding_dim=embedding_dim,
        _weight=features,
    )
    features.weight.requires_grad = False

    return features


def jarvis_load_atoms(path: Path):
    """Load Atoms data from individual files.

    Assume xyz and pdb formats correspond to non-periodic structures
    and add vacuum padding to the cell accordingly.
    """
    if path.suffix == ".vasp":
        atoms = jarvis.core.atoms.Atoms.from_poscar(path)
    elif path.suffix == ".cif":
        atoms = jarvis.core.atoms.Atoms.from_cif(path)
    elif path.suffix == ".xyz":
        atoms = jarvis.core.atoms.Atoms.from_xyz(path, box_size=500)
    else:
        raise NotImplementedError(f"{path.suffix} not currently supported.")

    return Atoms(atoms)


class Atoms:
    """Atoms: a basic crystal structure data class.

    ## What is fundamental about a crystal?

    - lattice (lattice parameters / cell matrix)
    - positions (fractional coordinates)
    - numbers (atom identities / fractional site occupancies)
    - periodic boundary conditions - 3D, 2D, ...?

    so work with this foundational representation, and transform it

    - DGLGraph
    - ALIGNNGraphTuple
    - PyG format
    - ...

    """

    @dispatch
    def __init__(
        self,
        lattice: torch.Tensor,
        positions: torch.Tensor,
        numbers: torch.Tensor,
        *,
        batch_num_atoms: Optional[Iterable[int]] = None,
    ):
        """Create atoms"""
        self.lattice = lattice
        self.positions = positions
        self.numbers = numbers
        self._batch_num_atoms = batch_num_atoms

    @dispatch
    def __init__(  # noqa: F811
        self, lattice: Iterable, positions: Iterable, numbers: Iterable
    ):
        dtype = torch.get_default_dtype()
        self.lattice = torch.tensor(lattice, dtype=dtype)
        self.positions = torch.tensor(positions, dtype=dtype)
        self.numbers = torch.tensor(numbers, dtype=Z_dtype)

    @dispatch
    def __init__(self, atoms: jarvis.core.atoms.Atoms):  # noqa: F811
        self.__init__(atoms.lattice.matrix, atoms.frac_coords, atoms.atomic_numbers)

    @dispatch
    def __init__(self, atoms: ase.Atoms):  # noqa: F811
        self.__init__(
            atoms.cell.array, atoms.get_scaled_positions(), atoms.get_atomic_numbers()
        )

    @property
    def batch_num_atoms(self):
        if self.batched():
            return self._batch_num_atoms
        return [self.positions.shape[0]]

    def batched(self) -> bool:
        return self.lattice.ndim == 3

    def __repr__(self):
        kws = [f"{key}={value!r}" for key, value in self.__dict__.items()]
        return "{}({})".format(type(self).__name__, ", ".join(kws))

    def __len__(self):
        """Check length of atoms if not batched."""
        if self.lattice.ndim == 2:
            return self.positions.shape[0]
        else:
            raise NotImplementedError(
                "__len__ not defined for batched atoms. use batch_num_atoms instead."
            )

    def to(self, device, non_blocking: bool = False):
        self.lattice = self.lattice.to(device, non_blocking=non_blocking)
        self.positions = self.positions.to(device, non_blocking=non_blocking)
        self.numbers = self.numbers.to(device, non_blocking=non_blocking)
        return self


def spglib_cell(x: Atoms):
    """Unpack Atoms to spglib tuple format."""
    if x.batched():
        return [spglib_cell(at) for at in unbatch(x)]
    return (x.lattice, x.positions, x.numbers)


@dispatch
def batch(atoms: List[Atoms]) -> Atoms:
    batch_num_atoms = [a.positions.shape[0] for a in atoms]
    lattice = torch.stack([a.lattice for a in atoms])
    numbers = torch.hstack([a.numbers for a in atoms])
    positions = torch.vstack([a.positions for a in atoms])
    return Atoms(lattice, positions, numbers, batch_num_atoms=batch_num_atoms)


@dispatch
def unbatch(atoms: Atoms) -> List[Atoms]:
    num_atoms = atoms.batch_num_atoms
    lattice = [lat for lat in atoms.lattice]
    positions = torch.split(atoms.positions, num_atoms)
    numbers = torch.split(atoms.numbers, num_atoms)
    return [Atoms(c, n, x) for c, n, x in zip(lattice, positions, numbers)]


AtomsGraph: TypeAlias = dgl.DGLGraph


@dispatch
def batch(g: List[AtomsGraph]):  # noqa: F811
    return dgl.batch(g)
