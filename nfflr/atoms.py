__all__ = ()

from pathlib import Path
from collections.abc import Iterable
from typing import Optional

import ase

try:
    import dgl
    from typing import TypeAlias

    _dgl_available = True
except ImportError:
    _dgl_available = False


import torch
from plum import dispatch

import jarvis.core.atoms

Z_dtype = torch.int


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
    """Atoms: basic atomistic data structure supporting batching

    The goal is to make it easy to work with a common representation,
    and transform it as needed to other formats.

    - DGLGraph
    - ALIGNNGraphTuple
    - PyG format
    - ...

    Parameters
    ----------
    cell : torch.Tensor
        cell matrix / lattice parameters
    positions : torch.Tensor
        Cartesian coordinates
    numbers : torch.Tensor
        atomic numbers

    Other Parameters
    ----------------
    batch_num_atoms : Iterable[int], optional
        batch size for struct-of-arrays batched Atoms


    """

    @dispatch
    def __init__(
        self,
        cell: torch.Tensor,
        positions: torch.Tensor,
        numbers: torch.Tensor,
        *,
        batch_num_atoms: Optional[Iterable[int]] = None,
    ):
        """Create atoms"""
        self.cell = cell
        self.positions = positions
        self.numbers = numbers
        self._batch_num_atoms = batch_num_atoms

    @dispatch
    def __init__(  # noqa: F811
        self, cell: Iterable, positions: Iterable, numbers: Iterable
    ):
        dtype = torch.get_default_dtype()
        if isinstance(cell, torch.Tensor):
            self.cell = cell if cell.dtype == dtype else cell.type(dtype)
        else:
            self.cell = torch.tensor(cell, dtype=dtype)
        self.positions = torch.tensor(positions, dtype=dtype)
        self.numbers = torch.tensor(numbers, dtype=Z_dtype)

    @dispatch
    def __init__(self, atoms: jarvis.core.atoms.Atoms):  # noqa: F811
        self.__init__(atoms.lattice.matrix, atoms.cart_coords, atoms.atomic_numbers)

    @dispatch
    def __init__(self, atoms: ase.Atoms):  # noqa: F811
        self.__init__(
            atoms.cell.array, atoms.get_positions(), atoms.get_atomic_numbers()
        )

    @property
    def batch_num_atoms(self):
        if self.batched():
            return self._batch_num_atoms
        return [self.positions.shape[0]]

    def batched(self) -> bool:
        """Determine if multiple structures are batched together."""
        return self.cell.ndim == 3

    def __repr__(self):
        kws = [f"{key}={value!r}" for key, value in self.__dict__.items()]
        return "{}({})".format(type(self).__name__, ", ".join(kws))

    def __len__(self):
        """Check length of atoms if not batched."""
        if self.cell.ndim == 2:
            return self.positions.shape[0]
        else:
            raise NotImplementedError(
                "__len__ not defined for batched atoms. use batch_num_atoms instead."
            )

    def scaled_positions(self) -> torch.Tensor:
        """Convert cartesian coordinates to fractional coordinates.

        Returns
        -------
        scaled_positions : torch.Tensor
        """
        return self.positions @ torch.linalg.inv(self.cell)

    def to(self, device, non_blocking: bool = False):
        """Transfer atoms data to compute device.

        Parameters
        ----------
        device : torch.device
            compute device, e.g. "cpu", "cuda", "mps"
        non_blocking : bool, optional
            attempt asynchronous transfer.
        """
        self.cell = self.cell.to(device, non_blocking=non_blocking)
        self.positions = self.positions.to(device, non_blocking=non_blocking)
        self.numbers = self.numbers.to(device, non_blocking=non_blocking)
        return self


def to_ase(at: Atoms):
    """Convert nfflr.Atoms to ase.Atoms."""
    return ase.Atoms(cell=at.cell, positions=at.positions, numbers=at.numbers, pbc=True)


def spglib_cell(x: Atoms):
    """Unpack Atoms to spglib tuple format."""
    if x.batched():
        return [spglib_cell(at) for at in unbatch(x)]
    return (x.cell, x.scaled_positions(), x.numbers)


@dispatch
def batch(atoms: list[Atoms]) -> Atoms:
    batch_num_atoms = [a.positions.shape[0] for a in atoms]
    cell = torch.stack([a.cell for a in atoms])
    numbers = torch.hstack([a.numbers for a in atoms])
    positions = torch.vstack([a.positions for a in atoms])
    return Atoms(cell, positions, numbers, batch_num_atoms=batch_num_atoms)


@dispatch
def unbatch(atoms: Atoms) -> list[Atoms]:
    num_atoms = atoms.batch_num_atoms
    cell = [c for c in atoms.cell]
    positions = torch.split(atoms.positions, num_atoms)
    numbers = torch.split(atoms.numbers, num_atoms)
    return [Atoms(c, n, x) for c, n, x in zip(cell, positions, numbers)]


if _dgl_available:
    AtomsGraph: TypeAlias = dgl.DGLGraph

    @dispatch
    def batch(g: list[AtomsGraph]):  # noqa: F811
        return dgl.batch(g)
