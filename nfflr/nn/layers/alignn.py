import dgl
import torch
from typing import Optional, Literal

from nfflr.nn import EdgeGatedGraphConv


class ALIGNNConv(torch.nn.Module):
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


class SparseALIGNNConv(torch.nn.Module):
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
