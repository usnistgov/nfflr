import torch
from typing import Literal

from nfflr.nn.layers.norm import Norm


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
