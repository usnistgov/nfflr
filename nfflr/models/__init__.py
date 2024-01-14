"""NFFLR models"""
__all__ = (
    "ALIGNNConfig",
    "ALIGNN",
    "SchNetConfig",
    "SchNet",
    "TersoffConfig",
    "Tersoff",
)

# graph neural network implementations
from .gnn.alignn import ALIGNN, ALIGNNConfig
from .gnn.alignn_ff import ALIGNNFF, ALIGNNFFConfig
from .gnn.schnet import SchNet, SchNetConfig

# classical potential implementations

from .classical.tersoff import TersoffConfig, Tersoff

# TODO: conditional dependencies for KeOps (doesn't currently support Windows)
# from .classical.lj import LJParams, LennardJones

# TODO: fix conditional dependency on torchcubicspline
# before automatically importing EAM
# from .classical.eam import EAM
