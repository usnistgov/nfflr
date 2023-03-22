"""Module to generate networkx graphs."""
import numpy as np
from jarvis.core.specie import chem_data, get_node_attributes

from jarvis.core.atoms import Atoms as jAtoms
from typing import Tuple, Literal, Dict

try:
    import torch
    import dgl
except ModuleNotFoundError as exp:
    print("dgl/torch/tqdm is not installed.", exp)

from nfflr.atoms import Atoms


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


class Standardize(torch.nn.Module):
    """Standardize atom_features: subtract mean and divide by std."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """Register featurewise mean and standard deviation."""
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, g: dgl.DGLGraph):
        """Apply standardization to atom_features."""
        g = g.local_var()
        h = g.ndata.pop("atom_features")
        g.ndata["atom_features"] = (h - self.mean) / self.std
        return g


def compute_bond_cosines(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    # bond_cosine = torch.arccos((torch.clamp(bond_cosine, -1, 1)))

    return {"h": bond_cosine}


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
    pg = graphs.Graph.atom_dgl_multigraph(
        a, use_canonize=True, compute_line_graph=False
    )
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

    X_src = cart_coords
    num_atoms = X_src.shape[0]

    # determine how many supercells are needed for the cutoff radius
    recp = 2 * torch.pi * torch.linalg.inv(lattice_mat).T
    recp_len = torch.tensor([i for i in (torch.sqrt(torch.sum(recp**2, dim=1)))])

    maxr = torch.ceil((cutoff + bond_tol) * recp_len / (2 * torch.pi))
    nmin = torch.floor(torch.min(frac_coords, dim=0)[0]) - maxr
    nmax = torch.ceil(torch.max(frac_coords, dim=0)[0]) + maxr

    # construct the supercell index list
    all_ranges = [torch.arange(x, y, dtype=precision) for x, y in zip(nmin, nmax)]
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
