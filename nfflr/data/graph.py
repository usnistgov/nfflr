"""Module to generate DGLGraphs."""
__all__ = ()

from typing import Tuple, Dict

import dgl
import dgl.function as fn

import torch
import einops
import numpy as np
import array_api_compat
from scipy import spatial

import nfflr


def sort_edges_by_dst(g: dgl.DGLGraph):
    """Sort edges by increasing dst id"""
    if g.num_edges() <= 1:
        return g

    src, dst = g.edges(form="uv")
    edge_order = torch.argsort(dst)

    g_sorted = dgl.graph((src[edge_order], dst[edge_order]))

    for key, value in g.ndata.items():
        g_sorted.ndata[key] = value

    g_sorted.edata["r"] = g.edata["r"][edge_order].contiguous()

    return g_sorted


def compute_bond_cosines(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # messages flow: `a -> b -> c`
    # displacements: `a <- b <- c`
    # use law of cosines to compute angles cosines
    # negate dst bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    r1 = edges.src["r"]
    r2 = -edges.dst["r"]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    # bond_cosine = torch.arccos((torch.clamp(bond_cosine, -1, 1)))

    return {"h": bond_cosine}


def compute_bond_cosines_coincident(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # edge attention graph edge: (k, i) -> (j, i)
    # messages flow: `k -> i <- j`
    # displacements: `k <- i -> j`
    # use law of cosines to compute angles cosines
    # cos(theta) = ik \dot ji / (||ki|| ||ji||)
    r_ki = edges.src["r"]
    r_ji = edges.dst["r"]
    bond_cosine = torch.sum(r_ki * r_ji, dim=1) / (
        torch.norm(r_ki, dim=1) * torch.norm(r_ji, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    # bond_cosine = torch.arccos((torch.clamp(bond_cosine, -1, 1)))

    return {"h": bond_cosine}


def pad_ghost_region(a: nfflr.Atoms, cutoff: float = 5):
    """Pad ghost atoms using supercell tiling method."""
    # build radius graph in supercell
    repeats = expand_supercell(a.cell, a.pbc, cutoff)
    image_ranges = [torch.arange(-n, n + 1) for n in repeats]
    cell_images = torch.cartesian_prod(*image_ranges)

    # images atoms xyz
    x_supercell = a.positions + (cell_images @ a.cell).unsqueeze(1)
    x_supercell = einops.rearrange(
        x_supercell, "images atoms xyz -> (images atoms) xyz"
    )

    # pairwise distances between atoms in (0,0,0) cell and atoms in all periodic images
    dist = torch.cdist(
        a.positions, x_supercell, compute_mode="donot_use_mm_for_euclid_dist"
    )

    neighbor_mask = (dist > 1e-5) & (dist <= cutoff)

    # get node indices for edgelist from neighbor mask
    # src, v = torch.where(neighbor_mask)
    id_atom, id_nbr_image = torch.where(neighbor_mask)

    # keep only unique images
    # v = torch.unique(v)

    # divmod to get cell and atom ids
    cell_ids = torch.div(id_nbr_image, len(a), rounding_mode="floor")
    nbr_ids = id_nbr_image % len(a)

    return cell_images[cell_ids], nbr_ids


def expand_supercell(cell, pbc, cutoff):
    """Calculate supercell expansion to contain the cutoff domain."""
    xp = array_api_compat.array_namespace(cell)
    lengths = xp.linalg.norm(xp.linalg.pinv(cell), axis=0)
    repeats = xp.where(pbc, xp.ceil(cutoff * lengths), 0.0)
    return repeats


def periodic_radius_graph(
    a: nfflr.Atoms, r: float = 5, bond_tol: float = 0.15, dtype=torch.float
) -> dgl.DGLGraph:
    """Build periodic radius graph for crystal.

    For a message passing graph, src is the neighbor id and dst is the central atom.
    The displacement vector should point from the central atom to the neighbor
    """
    # build radius graph in supercell
    repeats = expand_supercell(a.cell, a.pbc, r)
    image_ranges = [torch.arange(-n, n + 1) for n in repeats]
    cell_images = torch.cartesian_prod(*image_ranges)

    # images atoms xyz
    x_supercell = a.positions + (cell_images @ a.cell).unsqueeze(1)

    # pairwise distances between atoms in (0,0,0) cell and atoms in all periodic images
    # disable matmul implementation, numerical instabilities -> self interactions
    # from cdist docs: (B P M) * (B R M) -> (B P R)
    # (atoms xyz) x (images neighbors xyz) -> (images atoms neighbors)
    dist = torch.cdist(
        a.positions, x_supercell, compute_mode="donot_use_mm_for_euclid_dist"
    )
    neighbor_mask = (dist > 1e-5) & (dist <= r)

    # get node indices for edgelist from neighbor mask
    id_image, id_atom, id_nbr = torch.where(neighbor_mask)

    # messages propagate src -> dst
    # this means propagation from *neighbor* to *self*
    # displacement vectors should point to neighbor from self, opposite the message flow
    g = dgl.graph((id_nbr, id_atom), num_nodes=len(a))
    g.ndata["coord"] = torch.asarray(a.positions, dtype=dtype)
    g.ndata["atomic_number"] = torch.asarray(a.numbers, dtype=torch.int)
    g.edata["r"] = torch.asarray(
        x_supercell[id_image, id_nbr] - a.positions[id_atom], dtype=dtype
    )

    return g


def periodic_radius_graph_kdtree(
    a: nfflr.Atoms,
    r: float = 5,
    bond_tol: float = 0.15,
    dtype=torch.get_default_dtype(),
) -> dgl.DGLGraph:
    """Build periodic radius graph for crystal.

    Uses a supercell kd-tree for neighbor queries.
    """
    # build radius graph in supercell
    repeats = expand_supercell(a.cell, a.pbc, r)
    image_ranges = [torch.arange(-n, n + 1) for n in repeats]
    cell_images = torch.cartesian_prod(*image_ranges)

    # images atoms xyz
    x_supercell = a.positions + (cell_images @ a.cell).unsqueeze(1)
    x_supercell = einops.rearrange(
        x_supercell, "images atoms xyz -> (images atoms) xyz"
    )

    primary = spatial.KDTree(a.positions)
    tiled = spatial.KDTree(x_supercell)

    dist = primary.sparse_distance_matrix(tiled, r, output_type="coo_matrix")

    # get node indices for edgelist from sparse distance matrix
    # either KDTree or scipy csr nonzero resolves the ≈0 distance self-edges...
    id_atom, id_nbr_image = dist.nonzero()

    # index into tiled cell image index to atom ids
    id_nbr = id_nbr_image % len(a)

    g = dgl.graph((id_nbr, id_atom), num_nodes=len(a))
    g.ndata["coord"] = torch.asarray(a.positions, dtype=dtype)
    g.ndata["atomic_number"] = torch.asarray(a.numbers, dtype=torch.int)
    g.edata["r"] = torch.asarray(
        x_supercell[id_nbr_image] - a.positions[id_atom], dtype=dtype
    )

    return g


def periodic_adaptive_radius_graph(
    a: nfflr.Atoms, r: float = 5, bond_tol: float = 0.15, dtype=torch.float
) -> dgl.DGLGraph:
    """Build periodic radius graph for crystal.

    Sets the cutoff distance to sqrt(2) times the largest nearest neighbor distance.
    """
    # build radius graph in supercell
    repeats = expand_supercell(a.cell, a.pbc, r)
    image_ranges = [torch.arange(-n, n + 1) for n in repeats]
    cell_images = torch.cartesian_prod(*image_ranges)

    # images atoms xyz
    x_supercell = a.positions + (cell_images @ a.cell).unsqueeze(1)
    x_supercell = einops.rearrange(
        x_supercell, "images atoms xyz -> (images atoms) xyz"
    )

    # pairwise distances between atoms in (0,0,0) cell and atoms in all periodic images
    dist = torch.cdist(
        a.positions, x_supercell, compute_mode="donot_use_mm_for_euclid_dist"
    )

    # collect nearest neighbor distance
    # k = 2 because first neighbor is a self-interaction
    # this is filtered out in the neighbor_mask selection
    nearest_dist, _ = dist.kthvalue(k=2)

    cutoff = np.sqrt(2) * nearest_dist.max()
    neighbor_mask = (dist > 1e-5) & (dist <= cutoff)

    # get node indices for edgelist from neighbor mask
    id_atom, id_nbr_image = torch.where(neighbor_mask)
    id_nbr = id_nbr_image % len(a)

    # index into tiled cell image index to atom ids
    g = dgl.graph((id_nbr, id_atom), num_nodes=len(a))

    # messages propagate src -> dst
    # this means propagation from *neighbor* to *self*
    g.ndata["coord"] = torch.asarray(a.positions, dtype=dtype)
    g.ndata["atomic_number"] = torch.asarray(a.numbers, dtype=torch.int)
    g.edata["r"] = torch.asarray(
        x_supercell[id_nbr_image] - a.positions[id_atom], dtype=dtype
    )

    # cosine cutoff - smoothly go from one to zero in [0, cutoff] interval
    # consider alternative: HOOMD cutoff with inner radius of nearest_dist.max()?
    rnorm = torch.norm(g.edata["r"], dim=1)
    g.edata["cutoff_value"] = (1 + torch.cos(np.pi * rnorm / cutoff)) / 2

    return g


def periodic_kshell_graph(
    a: nfflr.Atoms,
    k: int = 12,
    r: float = 15.0,
    bond_tol: float = 0.15,
    dtype=torch.float,
) -> dgl.DGLGraph:
    """Build periodic k-shell graph for crystal."""
    # set up supercell distance query
    repeats = expand_supercell(a.cell, a.pbc, r)
    image_ranges = [torch.arange(-n, n + 1) for n in repeats]
    cell_images = torch.cartesian_prod(*image_ranges)

    # images atoms xyz
    x_supercell = a.positions + (cell_images @ a.cell).unsqueeze(1)
    x_supercell = einops.rearrange(
        x_supercell, "images atoms xyz -> (images atoms) xyz"
    )

    # pairwise distances between atoms in (0,0,0) cell
    # and atoms in all periodic images
    dist = torch.cdist(
        a.positions, x_supercell, compute_mode="donot_use_mm_for_euclid_dist"
    )

    # collect kth-nearest neighbor distance
    # topk: k = 13 because first neighbor is a self-interaction
    # this is filtered out in the neighbor_mask selection
    nbrdist, _ = dist.topk(k + 1, largest=False)
    k_dist = nbrdist[:, -1]

    # expand k-NN graph to include all atoms in the
    # neighbor shell of the twelfth neighbor
    # broadcast the <= along the src axis
    atol = 1e-5
    neighbor_mask = (dist > atol) & (dist < k_dist[:, None] + atol)

    # get node indices for edgelist from neighbor mask
    id_atom, id_nbr_image = torch.where(neighbor_mask)
    id_nbr = id_nbr_image % len(a)

    # index into tiled cell image index to atom ids
    g = dgl.graph((id_nbr, id_atom))

    # messages propagate src -> dst
    # this means propagation from *neighbor* to *self*
    g.ndata["coord"] = torch.asarray(a.positions, dtype=dtype)
    g.ndata["atomic_number"] = torch.asarray(a.numbers, dtype=torch.int)
    g.edata["r"] = torch.asarray(
        x_supercell[id_nbr_image] - a.positions[id_atom], dtype=dtype
    )

    return g


def periodic_knn_graph(
    a: nfflr.Atoms,
    k: int = 12,
    r: float = 5,
    bond_tol: float = 0.15,
    dtype=torch.float,
) -> dgl.DGLGraph:
    """Build periodic knn graph for crystal.

    this doesn't work quite the same as the alignn version, which is a k-shell graph
    that constructs the shell graph for the kth neighbor's shell.
    """
    # set up supercell distance query
    repeats = expand_supercell(a.cell, a.pbc, r)
    image_ranges = [torch.arange(-n, n + 1) for n in repeats]
    cell_images = torch.cartesian_prod(*image_ranges)

    # images atoms xyz
    x_supercell = a.positions + (cell_images @ a.cell).unsqueeze(1)
    x_supercell = einops.rearrange(
        x_supercell, "images atoms xyz -> (images atoms) xyz"
    )

    # pairwise distances between atoms in (0,0,0) cell
    # and atoms in all periodic images
    dist = torch.cdist(
        a.positions, x_supercell, compute_mode="donot_use_mm_for_euclid_dist"
    )

    # collect kth-nearest neighbor distance
    # topk: k = 13 because first neighbor is a self-interaction
    # this is filtered out in the neighbor_mask selection
    nbrdist, id_nbr_image = dist.topk(k + 1, largest=False)

    # skip the nearest neighbor, which is a self-interaction
    id_atom = einops.repeat(torch.arange(len(a)), "atoms -> atoms k", k=k)
    id_nbr_image = id_nbr_image[:, 1:]

    id_atom = id_atom.flatten()
    id_nbr_image = id_nbr_image.flatten()
    id_nbr = id_nbr_image % len(a)

    # messages propagate src -> dst
    # this means propagation from *neighbor* to *self*
    g = dgl.graph((id_nbr, id_atom))
    g.ndata["coord"] = torch.asarray(a.positions, dtype=dtype)
    g.ndata["atomic_number"] = torch.asarray(a.numbers, dtype=torch.int)
    g.edata["r"] = torch.asarray(
        x_supercell[id_nbr_image] - a.positions[id_atom], dtype=dtype
    )

    return g


def periodic_sann_graph(
    a: nfflr.Atoms,
    max_neighbors: int = 32,
    cutoff_radius: float = 10.0,
    bond_tol: float = 0.15,
    dtype=torch.get_default_dtype(),
):
    """Solid Angle Nearest Neighbor algorithm (10.1063/1.4729313).

    This implementation uses an eager kd-tree k-neighbor query against a tiled supercell
    to build a fixed-format neighborlist (sorted by the kdtree query)
    so that the SANN cutoff criterion can be vectorized over atoms.
    """
    # set up supercell distance query
    repeats = expand_supercell(a.cell, a.pbc, cutoff_radius)
    image_ranges = [torch.arange(-n, n + 1) for n in repeats]
    cell_images = torch.cartesian_prod(*image_ranges)

    # images atoms xyz
    x_supercell = a.positions + (cell_images @ a.cell).unsqueeze(1)
    x_supercell = einops.rearrange(
        x_supercell, "images atoms xyz -> (images atoms) xyz"
    )

    # find k nearest tiled points to query points x
    # this is nice because scipy sorts the points for us!
    # start at neighbor 2 since x is always in x_supercell
    tiled = spatial.KDTree(x_supercell)
    distance, ids = tiled.query(a.positions, k=range(2, 2 + max_neighbors))

    # vectorize evaluation of SANN criterion
    ms = np.arange(1, 1 + max_neighbors) - 2.0
    ms[:2] = 0.01  # mask with value to give large rcut
    rcut = distance.cumsum(axis=1) / ms
    sann_neighbormask = distance < rcut  # ~(d >= rcut)

    # there is an off-by-one error here sometimes?
    rcut = np.array(
        [rcut[idx, idy] for idx, idy in enumerate(sann_neighbormask.sum(1))]
    )

    # broadcast the comparison to rcut to index into ids
    # messages propagate src -> dst: propagation from *neighbor* to *self*
    id_atom, nbr_supercell = np.where(distance <= rcut[:, None])

    # index into knn neighbor ids -> atom ids
    id_nbr_image = ids[id_atom, nbr_supercell]
    id_nbr = id_nbr_image % len(a)

    g = dgl.graph((id_nbr, id_atom))
    g.ndata["coord"] = torch.asarray(a.positions, dtype=dtype)
    g.ndata["atomic_number"] = torch.asarray(a.numbers, dtype=torch.int)
    g.edata["r"] = torch.asarray(
        x_supercell[id_nbr_image] - a.positions[id_atom], dtype=dtype
    )

    g.ndata["cutoff_distance"] = torch.asarray(rcut, dtype=dtype)

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


def edge_coincidence_graph(g: dgl.DGLGraph, shared=False, cutoff: float | None = None):
    # get all pairs of incident edges for each node
    # torch.combinations gives half the pairs
    edgepairs = [
        torch.combinations(g.in_edges(id_node, form="eid"), r=2, with_replacement=False)
        for id_node in g.nodes()
    ]

    eids, ps = einops.pack(edgepairs, "* d")

    src, dst = eids.T

    # to_bidirected fills in the other half of edge pairs all at once
    # don't drop any bonds - make a graph with no edges if there are no triplets (?)
    t = dgl.to_bidirected(dgl.graph((src, dst), num_nodes=g.num_edges()))

    # TODO: return empty graph here if there are no triplets?
    # also TODO: use a heterograph to include self-interactions for the attention?
    # maybe this is more natural in KeOps...?

    if shared:
        t.ndata["r"] = g.edata["r"]

    if cutoff is not None:
        with torch.no_grad():
            t.apply_edges(fn.u_sub_v("r", "r", "d"))
            tripletmask = t.edata["d"].norm(dim=1) < cutoff
            del t.edata["d"]
            t = t.edge_subgraph(
                torch.arange(t.num_edges(), device=src.device)[tripletmask],
                relabel_nodes=False,
            )

    return t
