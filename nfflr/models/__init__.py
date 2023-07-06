"""NFFLR models"""

# graph neural network implementations
from .gnn.alignn import ALIGNN, ALIGNNConfig
from .gnn.alignn_ff import ALIGNNFF, ALIGNNFFConfig

# classical potential implementations
from .classical.lj import LJParams, LennardJones
from .classical.tersoff import TersoffConfig, Tersoff

# TODO: fix conditional dependency on torchcubicspline
# before automatically importing EAM
# from .classical.eam import EAM
