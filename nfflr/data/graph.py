"""Module to generate DGLGraphs."""
__all__ = ()

from typing import Tuple, Dict

import dgl
import torch
import numpy as np
from scipy import spatial

import nfflr


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
    a: nfflr.Atoms, r: float = 5, bond_tol: float = 0.15
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct supercell coordinate array with indices to atoms in (000) image."""
    n = len(a)
    Xcart = a.positions.clone().detach()
    Xfrac = a.scaled_positions().clone().detach()
    # Xfrac = Xcart @ torch.linalg.inv(a.cell) # this should be a function

    # cutoff -> calculate which periodic images to consider
    recp = 2 * np.pi * torch.linalg.inv(a.cell.T)
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
    X_supercell = (cell_images @ a.cell).unsqueeze(1) + Xcart
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


# def periodic_radius_graph_dgl(
#     a: nfflr.Atoms, r: float = 5, bond_tol: float = 0.15
# ) -> dgl.DGLGraph:
#     """Build periodic radius graph for crystal.

#     TODO: support 2D, 1D, or non-periodic boundary conditions
#     """
#     # X_supercell, root_ids = tile_supercell(a, r, bond_tol)
#     _, X_supercell, root_ids = tile_supercell_2(
#         a.positions.double(), a.cell.double(), r, bond_tol
#     )

#     # build radius graph in supercell
#     g = dgl.radius_graph(X_supercell, r)
#     g.ndata["Xcart"] = X_supercell.type(torch.get_default_dtype())

#     # reduce supercell graph to (000) image with periodic edges
#     g = reduce_supercell_graph(g, root_ids)

#     # add the fractional coordinates
#     # note: to do this differentiably, probably want to store
#     # fractional coordinates and cell matrix and compute cartesian coordinates
#     # and bond distances on demand?
#     g.ndata["Xfrac"] = a.positions

#     g.ndata["atomic_number"] = a.numbers.type(torch.int)

#     return g

# # tile periodic images into X_dst
# # index id_dst into X_dst maps to atom id as id_dest % num_atoms
# X_dst = (cell_images @ cell_matrix)[:, None, :] + X_src
# X_dst = X_dst.reshape(-1, 3)


def tile_supercell_2(
    x_cart, cell, r: float = 5, bond_tol: float = 0.15
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct supercell coordinate array with indices to atoms in (000) image."""

    # X_src = torch.tensor(a.cart_coords, dtype=precision)
    # lattice_matrix = torch.tensor(a.lattice_mat, dtype=precision)
    n, _ = x_cart.shape
    x_frac = x_cart @ torch.linalg.inv(cell)

    # cutoff -> calculate which periodic images to consider
    # recp_len = np.array(a.lattice.reciprocal_lattice().abc)
    recp = 2 * np.pi * torch.linalg.inv(cell.T)
    recp_len = torch.sqrt(torch.sum(recp**2, dim=1))

    # maxr =  np.ceil((r + bond_tol) * recp_len / (2 * np.pi))
    maxr = torch.ceil((r + bond_tol) * recp_len / (2 * np.pi))

    # nmin =  np.floor(np.min(a.frac_coords, axis=0)) - maxr
    nmin = torch.floor(x_frac.min(0).values) - maxr

    # nmax =  np.ceil(np.max(a.frac_coords, axis=0)) + maxr
    nmax = torch.ceil(x_frac.max(0).values) + maxr

    # all_ranges = [torch.arange(x, y, dtype=precision) for x, y in zip(nmin, nmax)]
    all_ranges = [
        torch.arange(x, y, dtype=x_frac.dtype) for x, y in zip(nmin, nmax, strict=True)
    ]

    # cell_images = torch.cartesian_prod(*all_ranges)
    cell_images = torch.cartesian_prod(*all_ranges)

    # get single image index for cell 000
    root_cell = (cell_images == 0).all(dim=1).nonzero().item()
    root_ids = n * root_cell + torch.arange(n)

    # tile periodic images into X_dst
    # index id_dst into X_dst maps to atom id as id_dest % num_atoms

    # X_dst = (cell_images @ lattice_matrix)[:, None, :] + X_src
    x_supercell = (cell_images @ cell).unsqueeze(1) + x_cart

    # X_dst = X_dst.reshape(-1, 3)
    x_supercell = x_supercell.reshape(-1, 3)

    return x_cart, x_supercell, root_ids, cell_images


def pad_ghost_region(atoms: nfflr.Atoms, cutoff: float = 5):
    """Pad ghost atoms using supercell tiling method."""

    # note: maybe don't need to do this in double precision?
    x_src, x_supercell, root_ids, offsets = tile_supercell_2(
        atoms.positions, atoms.cell, cutoff
    )

    x_supercell = x_supercell.reshape(-1, 3)  # (n_cells * n_atoms, n_dim)

    # pairwise distances between atoms in (0,0,0) cell
    # and atoms in all periodic images
    dist = torch.cdist(x_src, x_supercell)

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
    a: nfflr.Atoms, r: float = 5, bond_tol: float = 0.15, dtype=torch.float
) -> dgl.DGLGraph:
    """Build periodic radius graph for crystal.

    TODO: support 2D, 1D, or non-periodic boundary conditions
    """

    # build radius graph in supercell
    x_src, x_supercell, root_ids, cell_images = tile_supercell_2(
        a.positions.double(), a.cell.double(), r, bond_tol
    )

    # pairwise distances between atoms in (0,0,0) cell
    # and atoms in all periodic images
    dist = torch.cdist(x_src, x_supercell)

    atol = 1e-5
    neighbor_mask = (dist > atol) & (dist < r)

    # get node indices for edgelist from neighbor mask
    src, v = torch.where(neighbor_mask)

    # index into tiled cell image index to atom ids
    g = dgl.graph((src, v % len(a.numbers)))

    # add the fractional coordinates
    # note: to do this differentiably, probably want to store
    # fractional coordinates and cell matrix and compute cartesian coordinates
    # and bond distances on demand?
    # g.ndata["Xfrac"] = a.positions.to(dtype)

    # messages propagate src -> dst
    # this means propagation from *neighbor* to *self*
    g.ndata["coord"] = x_src.to(dtype)
    g.edata["r"] = (x_supercell[v] - x_src[src]).to(dtype)

    g.ndata["atomic_number"] = a.numbers.type(torch.int)

    return g


def periodic_radius_graph_kdtree(
    a: nfflr.Atoms, r: float = 5, bond_tol: float = 0.15
) -> dgl.DGLGraph:
    """Build periodic radius graph for crystal.

    TODO: support 2D, 1D, or non-periodic boundary conditions
    """

    # build radius graph in supercell
    x_src, x_supercell, root_ids, cell_images = tile_supercell_2(
        a.positions.double(), a.cell.double(), r, bond_tol
    )
    primary = spatial.KDTree(x_src)
    tiled = spatial.KDTree(x_supercell)
    dist = primary.sparse_distance_matrix(tiled, r, output_type="coo_matrix")

    # get node indices for edgelist from sparse distance matrix
    # either KDTree or scipy csr nonzero resolves the ≈0 distance self-edges...
    src, v = dist.nonzero()

    # index into tiled cell image index to atom ids
    g = dgl.graph((src, v % len(a.numbers)))

    # add the fractional coordinates
    # note: to do this differentiably, probably want to store
    # fractional coordinates and cell matrix and compute cartesian coordinates
    # and bond distances on demand?
    # g.ndata["Xfrac"] = a.positions

    g.ndata["coord"] = x_src.float()
    g.edata["r"] = (x_supercell[v] - x_src[src]).float()

    g.ndata["atomic_number"] = a.numbers.type(torch.int)

    return g


def periodic_adaptive_radius_graph(
    a: nfflr.Atoms, r: float = 5, bond_tol: float = 0.15, dtype=torch.float
) -> dgl.DGLGraph:
    """Build periodic radius graph for crystal.

    TODO: support 2D, 1D, or non-periodic boundary conditions
    """

    # build radius graph in supercell
    x_src, x_supercell, root_ids, cell_images = tile_supercell_2(
        a.positions.double(), a.cell.double(), r, bond_tol
    )

    # pairwise distances between atoms in (0,0,0) cell
    # and atoms in all periodic images
    dist = torch.cdist(x_src, x_supercell)

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
    # fractional coordinates and cell matrix and compute cartesian coordinates
    # and bond distances on demand?
    # g.ndata["Xfrac"] = a.positions.to(dtype)

    # messages propagate src -> dst
    # this means propagation from *neighbor* to *self*
    g.ndata["coord"] = x_src.to(dtype)
    g.edata["r"] = (x_supercell[v] - x_src[src]).to(dtype)

    # cosine cutoff - smoothly go from one to zero in [0, cutoff] interval
    # consider alternative: HOOMD cutoff with inner radius of nearest_dist.max()?
    rnorm = torch.norm(g.edata["r"], dim=1)
    g.edata["cutoff_value"] = (1 + torch.cos(np.pi * rnorm / cutoff)) / 2

    g.ndata["atomic_number"] = a.numbers.type(torch.int)

    return g


def periodic_kshell_graph(
    a: nfflr.Atoms,
    k: int = 12,
    r: float = 15.0,
    bond_tol: float = 0.15,
    dtype=torch.float,
) -> dgl.DGLGraph:
    """Build periodic radius graph for crystal.

    TODO: support 2D, 1D, or non-periodic boundary conditions
    """

    # build radius graph in supercell
    x_src, x_supercell, root_ids, cell_images = tile_supercell_2(
        a.positions.double(), a.cell.double(), r, bond_tol
    )

    # pairwise distances between atoms in (0,0,0) cell
    # and atoms in all periodic images
    dist = torch.cdist(x_src, x_supercell)

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
    # fractional coordinates and cell matrix and compute cartesian coordinates
    # and bond distances on demand?
    # g.ndata["Xfrac"] = a.positions.to(dtype)

    # messages propagate src -> dst
    # this means propagation from *neighbor* to *self*
    g.ndata["coord"] = x_src.to(dtype)
    g.edata["r"] = (x_supercell[v] - x_src[src]).to(dtype)

    g.ndata["atomic_number"] = a.numbers.type(torch.int)

    return g


def periodic_knn_graph(
    a: nfflr.Atoms, k: int = 12, r: float = 5, bond_tol: float = 0.15
) -> dgl.DGLGraph:
    """Build periodic knn graph for crystal.

    this doesn't work quite the same as the alignn version, which is a k-shell graph
    that constructs the shell graph for the kth neighbor's shell...

    TODO: support 2D, 1D, or non-periodic boundary conditions
    """
    x_supercell, root_ids = tile_supercell(a, r, bond_tol)

    # build knn graph in supercell
    g = dgl.knn_graph(x_supercell, k, exclude_self=True)
    g.ndata["Xcart"] = x_supercell.type(torch.get_default_dtype())

    # reduce supercell graph to (000) image with periodic edges
    g = reduce_supercell_graph(g, root_ids)

    # add the fractional coordinates
    # note: to do this differentiably, probably want to store
    # fractional coordinates and cell matrix and compute cartesian coordinates
    # and bond distances on demand?
    # g.ndata["Xfrac"] = a.positions @ torch.linalg.inv(a.cell)

    g.ndata["atomic_number"] = a.numbers.type(torch.int)

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
