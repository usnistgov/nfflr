import dgl
import torch
from torch import nn
from typing import Literal


class Norm(nn.Module):
    def __init__(
        self,
        num_features: int,
        norm_type: Literal["batchnorm", "layernorm", "instancenorm"] = "layernorm",
        mode: Literal["node", "edge"] = "node",
    ):
        super().__init__()
        self.norm_type = norm_type
        self.mode = mode

        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(num_features)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(num_features)
        elif norm_type == "instancenorm":
            self.norm = InstanceNorm(mode=mode)
            self.forward = self._forward_graph_instancenorm

    def forward(self, x: torch.Tensor):
        return self.norm(x)

    def _forward_graph_instancenorm(self, g: dgl.DGLGraph, x: torch.Tensor):
        return self.norm(g, x)


class InstanceNorm(nn.Module):
    def __init__(self, mode: Literal["node", "edge"] = "node", eps: float = 1e-6):
        super().__init__()
        self.mode = mode
        self.eps = eps

        if self.mode == "node":
            self.readout = dgl.readout.mean_nodes
            self.broadcast = dgl.broadcast_nodes
            self.data = lambda g: g.ndata

        elif self.mode == "edge":
            self.readout = dgl.readout.mean_edges
            self.broadcast = dgl.broadcast_edges
            self.data = lambda g: g.edata
        else:
            raise NotImplementedError(f"InstanceNorm(mode='{mode}') not supported")

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor):
        g = g.local_var()

        # compute per-instance channel-wise mean
        self.data(g)["_x"] = x
        mu = self.broadcast(g, self.readout(g, "_x"))

        # compute per-instance channel-wise variance
        self.data(g)["_sqdev"] = (x - mu) ** 2
        var = self.readout(g, "_sqdev")
        std = torch.sqrt(var + self.eps)

        # apply normalization
        return (x - mu) / self.broadcast(g, std)
