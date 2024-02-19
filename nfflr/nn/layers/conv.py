from typing import Literal

import torch
from torch.nn import functional as F

import dgl
import dgl.function as fn

# import nfflr
from nfflr.nn import Norm


class EdgeGatedGraphConv(torch.nn.Module):
    """Edge gated graph convolution

    See `Bresson and Laurent <https://arxiv.org/abs/1711.07553>`_
    :footcite:p:`bresson2018residual` for reference,
    and refer to `Dwivedi et al. <https://www.jmlr.org/papers/v24/22-0567.html>`_
    :footcite:p:`dwivedi2022` for detailed discussion.

    .. math ::
        x_i^{l+1} = SiLU ( U x_i^l + \sum_{j \in \mathcal{N}(i)} \eta_{ij} ⊙ V x_j^l)


    This is similar to the interaction from
    `CGCNN <https://dx.doi.org/10.1103/physrevlett.120.145301>`_ :footcite:p:`cgcnn`,
    but edge features only go into the soft attention / edge gating function,
    and the primary node update function is W cat(u, v) + b

    .. footbibliography::

    Parameters
    ----------
    input_features : int
    output_features : int
    residual : bool,  default=True
        add skip connection for both node and edge features
    norm : {"layernorm", "batchnorm", "instancenorm"}, optional
    skip_edgenorm : bool default=False
        omit normalization of edge features
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        residual: bool = True,
        norm: Literal["layernorm", "batchnorm", "instancenorm"] = "layernorm",
        skip_edgenorm: bool = False,
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()

        self.skip_edgenorm = skip_edgenorm

        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = torch.nn.Linear(input_features, output_features)
        self.dst_gate = torch.nn.Linear(input_features, output_features)
        self.edge_gate = torch.nn.Linear(input_features, output_features)

        if self.skip_edgenorm:
            self.norm_edges = torch.nn.Identity()
        else:
            self.norm_edges = Norm(output_features, norm, mode="edge")

        self.src_update = torch.nn.Linear(input_features, output_features)
        self.dst_update = torch.nn.Linear(input_features, output_features)
        self.norm_nodes = Norm(output_features, norm)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Edge-gated graph convolution.

        Parameters
        ----------
        g : dgl.DGLGraph
            input graph
        node_feats : torch.Tensor
            input node features
        edge_feats : torch.Tensor
            input edge features

        Returns
        -------
        node_features : torch.Tensor
        edge_features : torch.Tensor
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
