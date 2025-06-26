"""Shared model-building components."""
import torch
import numpy as np
from torch.autograd import grad

import dgl
import dgl.function as fn

from packaging import version

if version.parse(torch.__version__) < version.parse("2.0"):
    from functorch import vmap
else:
    from torch import vmap


class EnergyScaling(torch.nn.Module):
    r"""Energy unit scaling module.

    Shift and scale energy to standardize parameter gradients at initialization.

    We want to scale the energy units so either the forces ($\sigma^2_F$)
    or the atomwise energy ($\sigma^2_a$) have variance close to unity...?

    One perspective:
    try to keep the gradients through the body of the network on a uniform scale.
    With typical initialization strategies the atomwise energy contributions
    should have variance close to one at initialization.

    We can calculate the desired $\sigma_E = \sigma_a \sqrt{N_{avg}}$ based on this,
    where $N_{avg}$ is the average number of atoms in the training set configurations.

    Note: make sure model predictions are centered at zero at initialization.
    This might require re-centering the predictions by setting the output layer bias.
    """

    def __init__(self, energy: np.array, n_avg: float):
        super().__init__()

        # sigma_E = sigma_a / sqrt(n_avg)
        target_std = 1 * np.sqrt(n_avg) / np.sqrt(2)
        # scale by ratio of standard deviations
        scale = np.std(energy) / target_std

        self.register_buffer(
            "_mean", torch.tensor(energy.mean(), dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "_scale", torch.tensor(scale, dtype=torch.get_default_dtype())
        )

        self.forward = self.transform

    def transform(self, x):
        """Apply standardization transform"""
        return (x - self._mean) / self._scale

    def scale(self, x):
        """Apply scaling without zero-centering for forces."""
        return x / self._scale

    def inverse_transform(self, x):
        """Inverse transform to send data to physical units."""
        return x * self._scale + self._mean

    def unscale(self, x):
        """Inverse transform without centering to send forces to physical units."""
        return x * self._scale


def autograd_forces(
    total_energy: torch.tensor,
    displacement_vectors: torch.tensor,
    g: dgl.DGLGraph,
    energy_units="eV/atom",
    reduce=True,
    compute_virial=False,
):
    # potentially we only need to build the computational graph
    # for the forces at training time, so that we can compute
    # the gradient of the force (and virial) loss?
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
        return pairwise_forces, None

    if compute_virial:
        # without cell volume, can only compute un-normalized stresses
        # these are the per-bond virial contributions:
        # TODO: double check sign convention wrt edge direction...
        # forces_ji contribution
        virials = vmap(torch.outer)(pairwise_forces, displacement_vectors)
        g.edata["virials"] = virials
        virial = 0.5 * dgl.readout.sum_edges(g, "virials")

        # consider reverse force sum -forces_ji
        virials = vmap(torch.outer)(-pairwise_forces, -displacement_vectors)
        g.edata["virials"] = virials
        virial += 0.5 * dgl.readout.sum_edges(g, "virials")

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

    if compute_virial:
        return forces, virial

    return forces
