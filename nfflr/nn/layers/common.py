import torch
from typing import Literal, Optional

from nfflr.nn.layers.norm import Norm


class FeedForward(torch.nn.Module):
    """Two-layer feedforward network."""

    def __init__(
        self,
        d_in: int,
        d_hidden: Optional[int] = None,
        d_out: Optional[int] = None,
        norm: bool = False,
    ):
        """Doc for init"""
        super().__init__()
        if d_hidden is None:
            d_hidden = 4 * d_in

        if d_out is None:
            d_out = d_in

        self.project_hidden = torch.nn.Linear(d_in, d_hidden)
        if norm:
            self.norm = torch.nn.LayerNorm(d_hidden)
        else:
            self.norm = None
        self.project_out = torch.nn.Linear(d_hidden, d_out)

        self.reset_parameters()

    def forward(self, x: torch.Tensor):
        """Doc for forward."""
        x = torch.nn.functional.silu(self.project_hidden(x))
        if self.norm:
            x = self.norm(x)
        return self.project_out(x)
        return self.layers(x)

    def reset_parameters(self):

        torch.nn.init.kaiming_normal_(self.project_hidden.weight, nonlinearity="relu")
        torch.nn.init.zeros_(self.project_hidden.bias)

        torch.nn.init.kaiming_normal_(self.project_out.weight, nonlinearity="relu")
        torch.nn.init.zeros_(self.project_out.bias)


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
