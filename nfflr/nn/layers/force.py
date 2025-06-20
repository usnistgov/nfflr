import dgl
import torch

from packaging import version

if version.parse(torch.__version__) < version.parse("2.0"):
    from functorch import vmap
else:
    from torch import vmap

import nfflr
from nfflr.nn import FeedForward


class ForceAutogradHead(torch.nn.Module):
    def __init__(self, compute_virial: bool = True):
        super().__init__()
        self.compute_virial = compute_virial

    def forward(self, x: torch.Tensor, g: dgl.DGLGraph):
        """Compute forces with autograd.

        g.edata["r"] requires grad.
        """
        result = nfflr.autograd_forces(
            x,
            g.edata["r"],
            g,
            energy_units="eV",
            compute_virial=self.compute_virial,
        )

        if self.compute_virial:
            forces, virial = result
            return dict(forces=forces, virial=virial)

        return dict(forces=forces)


class ForcePredictionHead(torch.nn.Module):
    """Direct force prediction - GemNet-like approach.

    force_i = sum_j MLP(x_ji) * r_ji / norm(r_ji)
    """

    def __init__(self, d_in: int = 16, d_hidden: int = 32, compute_virial: bool = True):
        super().__init__()
        self.compute_virial = compute_virial

        self.feedforward = FeedForward(d_in, d_hidden, d_out=1)

    def forward(self, x_ji: torch.Tensor, g: dgl.DGLGraph):

        # x_ji: bond embedding
        force_magnitude = self.feedforward(x_ji)

        forces_ji = force_magnitude * g.edata["r"] / g.edata["r"].norm(1)

        # sum reduction - also reduce reaction forces over reverse edges
        rg = dgl.reverse(g)
        forces = dgl.ops.copy_e_sum(g, forces_ji) - dgl.ops.copy_e_sum(rg, forces_ji)

        if self.compute_virial:
            displacement_vectors = g.edata["r"]

            # these are the per-bond virial contributions: forces_ji contribution
            virials = vmap(torch.outer)(forces_ji, displacement_vectors)
            g.edata["virials"] = virials
            virial = 0.5 * dgl.readout.sum_edges(g, "virials")

            # consider reverse force sum -forces_ji
            virials = vmap(torch.outer)(-forces_ji, -displacement_vectors)
            g.edata["virials"] = virials
            virial += 0.5 * dgl.readout.sum_edges(g, "virials")

            return dict(forces=forces, virial=virial)

        return dict(forces=forces)
