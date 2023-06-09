"""Shared model-building components."""
from typing import Optional, Literal

import numpy as np

import torch
from torch import nn
from torch.autograd import grad
from torch.nn import functional as F

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
    # for the forces at training time, so that we can compute    # the gradient of the force (and stress) loss?
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


class ChebyshevExpansion(nn.Module):
    """Expand features in (-1, 1) interval with Chebyshev basis."""

    def __init__(self, basis_size: int):
        super().__init__()
        self.n = torch.arange(basis_size)

    def forward(self, x):
        """Trigonometric definition of Chebyshev polynomial basis for |x| \lq 1.

        Tn(cos(theta)) = cos(n theta)
        """
        return torch.cos(self.n[:, None] * torch.acos(x))


class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer("centers", torch.linspace(self.vmin, self.vmax, self.bins))

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale**2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(-self.gamma * (distance.unsqueeze(1) - self.centers) ** 2)


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        norm: Literal["layernorm", "batchnorm"] = "layernorm",
    ):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()

        if norm == "layernorm":
            Norm = nn.LayerNorm
        elif norm == "batchnorm":
            Norm = nn.BatchNorm1d

        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            Norm(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, norm, silu layer."""
        return self.layer(x)


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also https://www.jmlr.org/papers/v24/22-0567.html


    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        residual: bool = True,
        norm: Literal["layernorm", "batchnorm"] = "layernorm",
        skip_edgenorm: bool = False,
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()

        if norm == "layernorm":
            Norm = nn.LayerNorm
        elif norm == "batchnorm":
            Norm = nn.BatchNorm1d

        self.skip_edgenorm = skip_edgenorm

        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)

        if self.skip_edgenorm:
            self.norm_edges = nn.Identity()
        else:
            self.norm_edges = Norm(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.norm_nodes = Norm(output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))

        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        # if edge attributes have a cutoff function value
        # multiply the edge gate values with the cutoff value
        cutoff_value = g.edata.get("cutoff_value")

        if cutoff_value is not None:
            g.edata["sigma"] = torch.sigmoid(m) * cutoff_value.unsqueeze(1)
        else:
            g.edata["sigma"] = torch.sigmoid(m)

        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h"))
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # softmax version seems to perform slightly worse
        # that the sigmoid-gated version
        # compute node updates
        # Linear(u) + edge_gates ⊙ Linear(v)
        # g.edata["gate"] = edge_softmax(g, y)
        # g.ndata["h_dst"] = self.dst_update(node_feats)
        # g.update_all(fn.u_mul_e("h_dst", "gate", "m"), fn.sum("m", "h"))
        # x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        x = F.silu(self.norm_nodes(x))
        y = F.silu(self.norm_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        norm: Literal["layernorm", "batchnorm"] = "layernorm",
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features, norm=norm)
        self.edge_update = EdgeGatedGraphConv(
            out_features,
            out_features,
            norm=norm,
        )

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z)

        return x, y, z


class SparseALIGNNConv(nn.Module):
    """ALIGNN with sparser line graph"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        norm: Literal["layernorm", "batchnorm"] = "layernorm",
        skip_last_norm: bool = False,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        # handle residual manually for line graph
        self.node_update = EdgeGatedGraphConv(
            in_features, out_features, norm=norm, residual=True
        )
        self.edge_update = EdgeGatedGraphConv(
            out_features,
            out_features,
            norm=norm,
            residual=False,
            skip_edgenorm=skip_last_norm,
        )

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        y_mask: Optional[torch.Tensor] = None,
    ):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()

        # handle residual for line graph manually
        z_residual = z

        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y)

        # Edge-gated graph convolution update on crystal graph
        if y_mask is not None:
            y_update, z_update = self.edge_update(lg, m[y_mask], z)
            m[y_mask] += y_update
            y = m

        else:
            y_update, z_update = self.edge_update(lg, m, z)
            y = y_update + m

        z = z_update + z_residual

        return x, y, z
