import torch

import nfflr
from nfflr.data.graph import (
    periodic_radius_graph,
    periodic_adaptive_radius_graph,
    periodic_kshell_graph,
)


class PeriodicRadiusGraph(torch.nn.Module):
    """Periodic radius graph transform."""

    def __init__(self, cutoff: float = 5.0, dtype=torch.float):
        super().__init__()
        self.cutoff = cutoff
        self.dtype = dtype

    def forward(self, x: nfflr.Atoms):
        """Compute periodic radius graph."""
        return periodic_radius_graph(x, r=self.cutoff, dtype=self.dtype)


class PeriodicAdaptiveRadiusGraph(torch.nn.Module):
    """Adaptive periodic radius graph transform."""

    def __init__(self, cutoff: float = 5.0, dtype=torch.float):
        super().__init__()
        self.cutoff = cutoff
        self.dtype = dtype

    def forward(self, x: nfflr.Atoms):
        return periodic_adaptive_radius_graph(x, r=self.cutoff, dtype=self.dtype)


class PeriodicKShellGraph(torch.nn.Module):
    """Periodic k-neighbor shell graph construction.

    Parameters
    ----------
    k : int
        neighbor index defining radius of the shell graph
    cutoff : float
        maximum radial distance to consider
    dtype : torch.float
        dtype of the resulting graph features

    Returns
    -------
    dgl.DGLGraph

    """

    def __init__(self, k: int = 12, cutoff: float = 15.0, dtype=torch.float):
        super().__init__()
        self.k = k
        self.cutoff = cutoff
        self.dtype = dtype

    def forward(self, x: nfflr.Atoms):
        return periodic_kshell_graph(x, k=self.k, r=self.cutoff, dtype=self.dtype)
