import torch
from typing import Literal, Optional

from nfflr.nn.layers.norm import Norm


class FeedForward(torch.nn.Module):
    """Two-layer feedforward network."""

    def __init__(
        self, d_in: int, d_hidden: Optional[int] = None, d_out: Optional[int] = None
    ):
        """Doc for init"""
        super().__init__()
        if d_hidden is None:
            d_hidden = 4 * d_in

        if d_out is None:
            d_out = d_in

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor):
        """Doc for forward."""
        return self.layers(x)


class MLPLayer(torch.nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        norm: Literal["layernorm", "batchnorm"] = "layernorm",
    ):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()

        self.layer = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features),
            Norm(out_features, norm_type=norm),
            torch.nn.SiLU(),
        )

    def forward(self, x):
        """Linear, norm, silu layer."""
        return self.layer(x)
