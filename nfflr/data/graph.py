"""Module to generate DGLGraphs."""
__all__ = ()

from typing import Tuple, Literal, Dict

import dgl
import torch
import numpy as np
from scipy import spatial

from nfflr.data.atoms import Atoms


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


def tile_supercell(
    a: Atoms, r: float = 5, bond_tol: float = 0.15
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct supercell coordinate array with indices to atoms in (000) image."""
    n = len(a)
    Xfrac = a.positions.clone().detach()
    Xcart = Xfrac @ a.lattice

    # cutoff -> calculate which periodic images to consider
    recp = 2 * np.pi * torch.linalg.inv(a.lattice.T)
    recp_len = torch.sqrt(torch.sum(recp**2, dim=1))
    # recp_len = torch.tensor(a.lattice.reciprocal_lattice().abc)

    maxr = torch.ceil((r + bond_tol) * recp_len / (2 * np.pi))
    nmin = torch.floor(Xfrac.min(0).values) - maxr
    nmax = torch.ceil(Xfrac.max(0).values) + maxr
    all_ranges = [
        torch.arange(x, y, dtype=Xfrac.dtype) for x, y in zip(nmin, nmax, strict=True)
    ]
    cell_images = torch.cartesian_prod(*all_ranges)

    # get single image index for cell 000
    root_cell = (cell_images == 0).all(dim=1).nonzero().item()
    root_ids = n * root_cell + torch.arange(n)

    # tile periodic images into X_dst
    # index id_dst into X_dst maps to atom id as id_dest % num_atoms
    X_supercell = (cell_images @ a.lattice).unsqueeze(1) + Xcart
    X_supercell = X_supercell.reshape(-1, 3)

    return X_supercell, root_ids


def reduce_supercell_graph(g: dgl.DGLGraph, root_ids: torch.Tensor) -> dgl.DGLGraph:
    """Remove edges not involving (000) cell and remap node ids."""
    n = root_ids.shape[0]
    X_supercell = g.ndata["Xcart"]

    # keep edges with at least one node in root cell
    # note: edge_subgraph relabels the nodes
    src, dst = g.edges()
    edge_ids = torch.where(torch.isin(src, root_ids) | torch.isin(dst, root_ids))[0]
    g = dgl.edge_subgraph(g, edge_ids)

    # load coordinates based on original supercell indices
    g.ndata["Xcart"] = X_supercell[g.ndata["_ID"]]

    # compute all displacement vectors in the supercell subgraph
    g.apply_edges(dgl.function.v_sub_u("Xcart", "Xcart", "r"))

    # build new graph with same edge data but remap node ids into image (000)
    # src and dst use subset indices, not full supercell indices
    src, dst = g.edges()

    # map src and dst ids into periodic image (000)
    # first look up supercell indices with _ID values for src and dst
    src = g.ndata["_ID"][src]
    dst = g.ndata["_ID"][dst]

    # mod n to get original indices
    periodic_graph = dgl.graph((src % n, dst % n))
    periodic_graph.ndata["Xcart"] = X_supercell[root_ids]
    periodic_graph.edata["r"] = g.edata["r"]

    return periodic_graph


def periodic_radius_graph_dgl(
    a: Atoms, r: float = 5, bond_tol: float = 0.15
) -> dgl.DGLGraph:
    """Build periodic radius graph for crystal.

    TODO: support 2D, 1D, or non-periodic boundary conditions
    """
    # X_supercell, root_ids = tile_supercell(a, r, bond_tol)
    _, X_supercell, root_ids = tile_supercell_2(
        a.positions.double(), a.lattice.double(), r, bond_tol
    )

    # build radius graph in supercell
    g = dgl.radius_graph(X_supercell, r)
    g.ndata["Xcart"] = X_supercell.type(torch.get_default_dtype())

    # reduce supercell graph to (000) image with periodic edges
    g = reduce_supercell_graph(g, root_ids)

    # add the fractional coordinates
    # note: to do this differentiably, probably want to store
    # fractional coordinates and lattice matrix and compute cartesian coordinates
    # and bond distances on demand?
    g.ndata["Xfrac"] = a.positions

    g.ndata["atomic_number"] = a.numbers.type(torch.int)

    return g

    # # tile periodic images into X_dst
    # # index id_dst into X_dst maps to atom id as id_dest % num_atoms
    # X_dst = (cell_images @ lattice_matrix)[:, None, :] + X_src
    # X_dst = X_dst.reshape(-1, 3)


def tile_supercell_2(
    Xfrac, lattice, r: float = 5, bond_tol: float = 0.15
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct supercell coordinate array with indices to atoms in (000) image."""

    # X_src = torch.tensor(a.cart_coords, dtype=precision)
    # lattice_matrix = torch.tensor(a.lattice_mat, dtype=precision)
    n, _ = Xfrac.shape
    Xcart = Xfrac @ lattice

    # cutoff -> calculate which periodic images to consider
    # recp_len = np.array(a.lattice.reciprocal_lattice().abc)
    recp = 2 * np.pi * torch.linalg.inv(lattice.T)
    recp_len = torch.sqrt(torch.sum(recp**2, dim=1))

    # maxr =  np.ceil((r + bond_tol) * recp_len / (2 * np.pi))
    maxr = torch.ceil((r + bond_tol) * recp_len / (2 * np.pi))

    # nmin =  np.floor(np.min(a.frac_coords, axis=0)) - maxr
    nmin = torch.floor(Xfrac.min(0).values) - maxr

    # nmax =  np.ceil(np.max(a.frac_coords, axis=0)) + maxr
    nmax = torch.ceil(Xfrac.max(0).values) + maxr

    # all_ranges = [torch.arange(x, y, dtype=precision) for x, y in zip(nmin, nmax)]
    all_ranges = [
        torch.arange(x, y, dtype=Xfrac.dtype) for x, y in zip(nmin, nmax, strict=True)
    ]

    # cell_images = torch.cartesian_prod(*all_ranges)
    cell_images = torch.cartesian_prod(*all_ranges)

    # get single image index for cell 000
    root_cell = (cell_images == 0).all(dim=1).nonzero().item()
    root_ids = n * root_cell + torch.arange(n)

    # tile periodic images into X_dst
    # index id_dst into X_dst maps to atom id as id_dest % num_atoms

    # X_dst = (cell_images @ lattice_matrix)[:, None, :] + X_src
    X_supercell = (cell_images @ lattice).unsqueeze(1) + Xcart

    # X_dst = X_dst.reshape(-1, 3)
    X_supercell = X_supercell.reshape(-1, 3)

    return Xcart, X_supercell, root_ids, cell_images


def pad_ghost_region(atoms: Atoms, cutoff: float = 5):
    """Pad ghost atoms using supercell tiling method."""

    # note: maybe don't need to do this in double precision?
    X_src, X_supercell, root_ids, offsets = tile_supercell_2(
        atoms.positions, atoms.lattice, cutoff
    )

    X_supercell = X_supercell.reshape(-1, 3)  # (n_cells * n_atoms, n_dim)

    # pairwise distances between atoms in (0,0,0) cell
    # and atoms in all periodic images
    dist = torch.cdist(X_src, X_supercell)

    atol = 1e-5
    neighbor_mask = (dist > atol) & (dist < cutoff)

    # get node indices for edgelist from neighbor mask
    src, v = torch.where(neighbor_mask)

    # keep only unique images
    v = torch.unique(v)

    # divmod to get cell and atom ids
    cell_ids = torch.div(v, len(atoms), rounding_mode="floor")
    atom_ids = v % len(atoms)

    return offsets[cell_ids], atom_ids


def periodic_radius_graph(
    a: Atoms, r: float = 5, bond_tol: float = 0.15, dtype=torch.float
) -> dgl.DGLGraph:
    """Build periodic radius graph for crystal.

    TODO: support 2D, 1D, or non-periodic boundary conditions
    """

    # build radius graph in supercell
    X_src, X_supercell, root_ids, cell_images = tile_supercell_2(
        a.positions.double(), a.lattice.double(), r, bond_tol
    )

    # pairwise distances between atoms in (0,0,0) cell
    # and atoms in all periodic images
    dist = torch.cdist(X_src, X_supercell)

    atol = 1e-5
    neighbor_mask = (dist > atol) & (dist < r)

    # get node indices for edgelist from neighbor mask
    src, v = torch.where(neighbor_mask)

    # index into tiled cell image index to atom ids
    g = dgl.graph((src, v % len(a.numbers)))

    # add the fractional coordinates
    # note: to do this differentiably, probably want to store
    # fractional coordinates and lattice matrix and compute cartesian coordinates
    # and bond distances on demand?
    g.ndata["Xfrac"] = a.positions.to(dtype)

    # messages propagate src -> dst
    # this means propagation from *neighbor* to *self*
    g.ndata["coord"] = X_src.to(dtype)
    g.edata["r"] = (X_supercell[v] - X_src[src]).to(dtype)

    g.ndata["atomic_number"] = a.numbers.type(torch.int)

    return g


def periodic_radius_graph_kdtree(
    a: Atoms, r: float = 5, bond_tol: float = 0.15
) -> dgl.DGLGraph:
    """Build periodic radius graph for crystal.

    TODO: support 2D, 1D, or non-periodic boundary conditions
    """

    # build radius graph in supercell
    X_src, X_supercell, root_ids, cell_images = tile_supercell_2(
        a.positions.double(), a.lattice.double(), r, bond_tol
    )
    primary = spatial.KDTree(X_src)
    tiled = spatial.KDTree(X_supercell)
    dist = primary.sparse_distance_matrix(tiled, r, output_type="coo_matrix")

    # get node indices for edgelist from sparse distance matrix
    # either KDTree or scipy csr nonzero resolves the ≈0 distance self-edges...
    src, v = dist.nonzero()

    # index into tiled cell image index to atom ids
    g = dgl.graph((src, v % len(a.numbers)))

    # add the fractional coordinates
    # note: to do this differentiably, probably want to store
    # fractional coordinates and lattice matrix and compute cartesian coordinates
    # and bond distances on demand?
    g.ndata["Xfrac"] = a.positions

    g.ndata["coord"] = X_src.float()
    g.edata["r"] = (X_supercell[v] - X_src[src]).float()

    g.ndata["atomic_number"] = a.numbers.type(torch.int)

    return g


def periodic_adaptive_radius_graph(
    a: Atoms, r: float = 5, bond_tol: float = 0.15, dtype=torch.float
) -> dgl.DGLGraph:
    """Build periodic radius graph for crystal.

    TODO: support 2D, 1D, or non-periodic boundary conditions
    """

    # build radius graph in supercell
    X_src, X_supercell, root_ids, cell_images = tile_supercell_2(
        a.positions.double(), a.lattice.double(), r, bond_tol
    )

    # pairwise distances between atoms in (0,0,0) cell
    # and atoms in all periodic images
    dist = torch.cdist(X_src, X_supercell)

    # collect nearest neighbor distance
    # k = 2 because first neighbor is a self-interaction
    # this is filtered out in the neighbor_mask selection
    nearest_dist, _ = dist.kthvalue(k=2)

    cutoff = np.sqrt(2) * nearest_dist.max()

    atol = 1e-5
    neighbor_mask = (dist > atol) & (dist < cutoff)

    # get node indices for edgelist from neighbor mask
    src, v = torch.where(neighbor_mask)

    # index into tiled cell image index to atom ids
    g = dgl.graph((src, v % len(a.numbers)))

    # add the fractional coordinates
    # note: to do this differentiably, probably want to store
    # fractional coordinates and lattice matrix and compute cartesian coordinates
    # and bond distances on demand?
    g.ndata["Xfrac"] = a.positions.to(dtype)

    # messages propagate src -> dst
    # this means propagation from *neighbor* to *self*
    g.ndata["coord"] = X_src.to(dtype)
    g.edata["r"] = (X_supercell[v] - X_src[src]).to(dtype)

    # cosine cutoff - smoothly go from one to zero in [0, cutoff] interval
    # consider alternative: HOOMD cutoff with inner radius of nearest_dist.max()?
    rnorm = torch.norm(g.edata["r"], dim=1)
    g.edata["cutoff_value"] = (1 + torch.cos(np.pi * rnorm / cutoff)) / 2

    g.ndata["atomic_number"] = a.numbers.type(torch.int)

    return g


def periodic_kshell_graph(
    a: Atoms, k: int = 12, r: float = 15.0, bond_tol: float = 0.15, dtype=torch.float
) -> dgl.DGLGraph:
    """Build periodic radius graph for crystal.

    TODO: support 2D, 1D, or non-periodic boundary conditions
    """

    # build radius graph in supercell
    X_src, X_supercell, root_ids, cell_images = tile_supercell_2(
        a.positions.double(), a.lattice.double(), r, bond_tol
    )

    # pairwise distances between atoms in (0,0,0) cell
    # and atoms in all periodic images
    dist = torch.cdist(X_src, X_supercell)

    atol = 1e-5
    # neighbor_mask = (dist > atol) & (dist < r)

    # collect 12th-nearest neighbor distance
    # topk: k = 13 because first neighbor is a self-interaction
    # this is filtered out in the neighbor_mask selection
    nbrdist, _ = dist.topk(13, largest=False)
    k_dist = nbrdist[:, -1]

    # expand k-NN graph to include all atoms in the
    # neighbor shell of the twelfth neighbor
    # broadcast the <= along the src axis
    neighbor_mask = (dist > atol) & (dist < k_dist[:, None] + atol)

    # get node indices for edgelist from neighbor mask
    src, v = torch.where(neighbor_mask)

    # index into tiled cell image index to atom ids
    g = dgl.graph((src, v % len(a.numbers)))

    # add the fractional coordinates
    # note: to do this differentiably, probably want to store
    # fractional coordinates and lattice matrix and compute cartesian coordinates
    # and bond distances on demand?
    g.ndata["Xfrac"] = a.positions.to(dtype)

    # messages propagate src -> dst
    # this means propagation from *neighbor* to *self*
    g.ndata["coord"] = X_src.to(dtype)
    g.edata["r"] = (X_supercell[v] - X_src[src]).to(dtype)

    g.ndata["atomic_number"] = a.numbers.type(torch.int)

    return g


def periodic_knn_graph(
    a: Atoms, k: int = 12, r: float = 5, bond_tol: float = 0.15
) -> dgl.DGLGraph:
    """Build periodic knn graph for crystal.

    this doesn't work quite the same as the alignn version, which is a k-shell graph
    that constructs the shell graph for the kth neighbor's shell...

    TODO: support 2D, 1D, or non-periodic boundary conditions
    """
    X_supercell, root_ids = tile_supercell(a, r, bond_tol)

    # build knn graph in supercell
    g = dgl.knn_graph(X_supercell, k, exclude_self=True)
    g.ndata["Xcart"] = X_supercell.type(torch.get_default_dtype())

    # reduce supercell graph to (000) image with periodic edges
    g = reduce_supercell_graph(g, root_ids)

    # add the fractional coordinates
    # note: to do this differentiably, probably want to store
    # fractional coordinates and lattice matrix and compute cartesian coordinates
    # and bond distances on demand?
    g.ndata["Xfrac"] = a.positions

    g.ndata["atomic_number"] = a.numbers.type(torch.int)

    return g


### old stuff


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
