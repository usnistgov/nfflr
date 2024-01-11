"""Shared model-building components."""
import torch
from torch.autograd import grad

import dgl
import dgl.function as fn

from packaging import version

if version.parse(torch.__version__) < version.parse("2.0"):
    from functorch import vmap
else:
    from torch import vmap


def autograd_forces(
    total_energy: torch.tensor,
    displacement_vectors: torch.tensor,
    g: dgl.DGLGraph,
    energy_units="eV/atom",
    reduce=True,
    compute_stress=False,
):
    # potentially we only need to build the computational graph
    # for the forces at training time, so that we can compute
    # the gradient of the force (and stress) loss?
    create_graph = True

    # energy gradient contribution of each bond
    # dU/dr
    dy_dr = grad(
        total_energy,
        displacement_vectors,
        grad_outputs=torch.ones_like(total_energy),
        create_graph=create_graph,
        retain_graph=True,
    )[0]

    # forces: negative energy gradient -dU/dr
    pairwise_forces = -dy_dr

    if not reduce:
        return pairwise_forces

    if compute_stress:
        # without cell volume, can only compute un-normalized stresses
        # these are the per-bond stress contributions:
        # TODO: double check sign convention wrt edge direction...
        stresses = vmap(torch.outer)(-pairwise_forces, displacement_vectors)
        g.edata["stresses"] = stresses
        stress = dgl.readout.sum_edges(g, "stresses")

    # reduce over bonds to get forces on each atom
    g.edata["pairwise_forces"] = pairwise_forces
    g.update_all(fn.copy_e("pairwise_forces", "m"), fn.sum("m", "forces_ji"))

    # reduce over reverse edges too!
    rg = dgl.reverse(g, copy_edata=True)
    rg.update_all(fn.copy_e("pairwise_forces", "m"), fn.sum("m", "forces_ij"))

    forces = torch.squeeze(g.ndata["forces_ji"] - rg.ndata["forces_ij"])

    # if training against reduced energies, correct the force predictions
    if energy_units == "eV/atom":
        # broadcast |v(g)| across forces to under per-atom energy scaling

        n_nodes = torch.cat(
            [i * torch.ones(i, device=g.device) for i in g.batch_num_nodes()]
        )

        forces = forces * n_nodes[:, None]

    if compute_stress:
        return forces, stress

    return forces
