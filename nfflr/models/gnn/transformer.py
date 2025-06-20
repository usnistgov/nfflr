from plum import dispatch
from typing import Optional, Literal, Callable
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch import nn

import dgl
import dgl.function as fn
from dgl.nn import AvgPooling, SumPooling
from dgl.nn.functional import edge_softmax

import pykeops
from pykeops.torch import LazyTensor

import nfflr
from nfflr.nn import (
    RBFExpansion,
    ChebyshevExpansion,
    MLPLayer,
    AttributeEmbedding,
)


def _find_reverse_edge_ids(us, vs, rs):
    """Brute force edge reversal

    given edges (u, v) with displacement vectors r
    search for index of (v, u) with displacement -r
    """
    # N: number of edges
    N, D = rs.shape

    # KeOps can't handle int64
    _us = us.float()
    _vs = vs.float()

    # node labels corresponding to edges
    u_i = LazyTensor(_us.view(N, 1, 1))
    u_j = LazyTensor(_us.view(1, N, 1))
    v_i = LazyTensor(_vs.view(N, 1, 1))
    v_j = LazyTensor(_vs.view(1, N, 1))

    # displacement vectors
    r_i = LazyTensor(rs.view(N, 1, D))
    r_j = LazyTensor(rs.view(1, N, D))

    # match u_i == v_j
    id_mask = (u_i - v_j).abs() + (v_i - u_j).abs()

    # r_j should exactly equal -r_i
    # r_j + r_i = 0
    d_ji = (r_j + r_i).abs().sum(-1)

    return (d_ji + id_mask).argmin(0).squeeze()


def edge_attention_graph(g, shared: bool = False):
    lg = g.line_graph(backtracking=False, shared=False)

    # node ids are bond ids
    srcbond, dstbond = lg.edges()

    # so look up reverse edges of dstbond - this might be messy for periodic systems...
    # ids should be swapped and displacement vectors should point in opposite directions

    # one idea: sort the graph edges somehow beforehand to make search easier?
    # could use KeOps to do this search...

    src, dst = g.edges(form="uv")
    edgeperm = _find_reverse_edge_ids(src, dst, g.edata["r"])

    t = dgl.graph((srcbond, edgeperm[dstbond]))

    if shared:
        t.ndata["r"] = g.edata["r"]

    return t


@dataclass
class TFMConfig:
    """Hyperparameter schema for nfflr.models.gnn.tfm"""

    transform: Callable = nfflr.nn.PeriodicRadiusGraph(cutoff=5.0)
    cutoff: torch.nn.Module = nfflr.nn.Cosine(5.0)
    layers: int = 3
    norm: Literal["layernorm", "instancenorm"] = "layernorm"
    atom_features: str = "embedding"
    d_model: int = 256
    d_message: int = 64
    output_features: int = 1
    compute_forces: bool = False
    energy_units: Literal["eV", "eV/atom"] = "eV/atom"
    reference_energies: Optional[Literal["fixed", "trainable"]] = None


class TersoffAttention(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.channels = channels
        self.angle_encoder = ChebyshevExpansion(channels)
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, channels)))

    def forward(self, g: dgl.DGLGraph, t: dgl.DGLGraph, xij: torch.Tensor):

        # Tersoff attention
        t.ndata["xij"] = xij

        # compute bond angle embedding
        t.ndata["rnorm"] = -t.ndata["r"] / torch.norm(t.ndata["r"], dim=1)[:, None]
        t.apply_edges(fn.u_dot_v("rnorm", "rnorm", "cos_jik"))
        cos_jik = t.edata["cos_jik"]

        # use angle cosine as attention bias? like in ALiBi...
        # or use angle cosine embeddings?
        # with multihead attention, could use relative difference in angle
        # from some set of reference angles...?
        z_jik = self.angle_encoder(cos_jik)

        #
        t.apply_edges(fn.u_add_v("xij", "xij", "x_jik"))
        e_jik = F.silu(z_jik + t.edata.pop("x_jik"))

        a = (e_jik * self.attn).sum(dim=-1).unsqueeze(dim=2)  # (num_edge, num_heads, 1)

        t.edata["a"] = edge_softmax(t, a)

        g.update_all(fn.u_mul_e("xij", "a", "m"), fn.sum("m", "ft"))

        return g


class TersoffBlock(nn.Module):
    def __init__(self, d_model=256, d_message=64):
        super().__init__()
        self.attention = TersoffAttention()
        self.project_src = nn.Linear(d_model, d_message)
        self.project_dst = nn.Linear(d_model, d_message)
        self.project_edge = nn.Linear(d_model, d_message)

        self.feedforward = nfflr.nn.FeedForward(d_message, 4 * d_model, d_model)

    def forward(
        self, g: dgl.DGLGraph, t: dgl.DGLGraph, x: torch.Tensor, y: torch.Tensor
    ):
        g = g.local_var()
        t = t.local_var()

        # project down
        g.ndata["xj"] = self.project_src(x)
        g.ndata["xi"] = self.project_dst(x)
        g.apply_edges(fn.u_add_v("xj", "xi", "xij"))
        xij = g.edata.pop("xij") + self.project_edge(y)

        # Dwivedi et al apply the nonlinearity and edge residual
        # before the attention process
        # y = y + F.silu(xij)

        # attention layer
        x = self.attention(g, t, xij)

        # feedforward
        x = self.feedforward(x)

        return x


class TFM(nn.Module):
    """Prototype graph transformer model."""

    def __init__(self, config: TFMConfig = TFMConfig()):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        self.config = config
        self.transform = config.transform

        if config.atom_features == "embedding":
            self.atom_embedding = nfflr.nn.PeriodicTableEmbedding(config.d_model)
        else:
            self.atom_embedding = nfflr.nn.AttributeEmbedding(
                config.atom_features, config.d_model
            )

        self.bond_encoder = RBFExpansion(vmin=0, vmax=8.0, bins=config.d_model)

        if config.reference_energies is not None:
            self.reference_energy = nfflr.nn.AtomicReferenceEnergy(
                requires_grad=config.reference_energies == "trainable"
            )

        self.layers = nn.ModuleList(
            [
                TersoffBlock(config.d_model, config.d_message)
                for idx in range(config.layers)
            ]
        )

        if self.config.energy_units == "eV/atom":
            self.readout = AvgPooling()
        elif self.config.energy_units == "eV":
            self.readout = SumPooling()

        self.fc = nn.Linear(config.d_model, config.output_features)

        self.reset_atomic_reference_energies()

    def reset_atomic_reference_energies(self, values: Optional[torch.Tensor] = None):
        if hasattr(self, "reference_energy"):
            self.reference_energy.reset_parameters(values=values)

    @dispatch
    def forward(self, x):
        print("convert")
        return self.forward(nfflr.Atoms(x))

    @dispatch
    def forward(self, x: nfflr.Atoms):
        device = next(self.parameters()).device
        return self.forward(self.transform(x).to(device))

    @dispatch
    def forward(
        self,
        g: dgl.DGLGraph,
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata)
        """
        # print("forward")
        config = self.config
        g = g.local_var()

        # to compute forces, take gradient wrt g.edata["r"]
        # need to add bond vectors to autograd graph
        if config.compute_forces:
            g.edata["r"].requires_grad_(True)

        # edge coincidence graph
        t = edge_attention_graph(g, shared=True)

        # initial node features: atom feature network...
        atomic_number = g.ndata.pop("atomic_number").int()
        x = self.atom_embedding(atomic_number)

        # initial bond features
        bondlength = torch.norm(g.edata["r"], dim=1)
        y = self.bond_encoder(bondlength)
        g.edata["y"] = y

        if config.cutoff is not None:
            # save cutoff function value for application in EdgeGatedGraphconv
            g.edata["cutoff_value"] = self.config.cutoff(bondlength)

        # gated GCN updates: update node, edge features
        for layer in self.layers:
            x = layer(g, t, x, y)

        # predict per-atom energy contribution (in eV)
        atomwise_energy = self.fc(x)
        if hasattr(self, "reference_energy"):
            atomwise_energy += self.reference_energy(atomic_number)

        # total energy prediction
        # if config.energy_units = eV/atom, mean reduction
        output = torch.squeeze(self.readout(g, atomwise_energy))

        if config.compute_forces:
            forces, virial = nfflr.autograd_forces(
                output,
                g.edata["r"],
                g,
                energy_units=config.energy_units,
                compute_virial=True,
            )

            return dict(energy=output, forces=forces, virial=virial)

        return output
