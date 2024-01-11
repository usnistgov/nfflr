import torch
import numpy as np
from typing import Optional, Literal


class ChebyshevExpansion(torch.nn.Module):
    """Expand features in (-1, 1) interval with Chebyshev basis."""

    def __init__(self, basis_size: int):
        super().__init__()
        self.n = torch.arange(1, 1 + basis_size)

    def forward(self, x):
        """Trigonometric definition of Chebyshev polynomial basis for |x| \lq 1.

        Tn(cos(theta)) = cos(n theta)
        """
        return torch.cos(self.n * torch.acos(x).unsqueeze(1))


class RBFExpansion(torch.nn.Module):
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
