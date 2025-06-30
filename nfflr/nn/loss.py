import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Callable
from functools import partial

import torch
import einops
import numpy as np
import ignite.metrics

from nfflr.train.evaluation import pseudolog10


class RegularizationLoss(torch.nn.Module):
    """Loss stub for regularization term returned by a model."""

    def forward(self, inputs, target):
        assert target is None
        return inputs


class MeanAbsLogAccuracyRatio(torch.nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        err = torch.log(self.eps + inputs) - torch.log(self.eps + targets)
        return err.abs().mean()


class MeanSquaredLogAccuracyRatio(torch.nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        err = torch.log(self.eps + inputs) - torch.log(self.eps + targets)
        return err.square().mean()


class MultitaskForceFieldLoss(torch.nn.Module):
    """Multitask loss function wrapper for force field training.

    `inputs` and `targets` should be `Dict[str, torch.Tensor]`
    `targets` should contain `"n_atoms"`, the batched system size

    `tasks` can be an `Iterable[str]`, as in `["energy", "forces"]`
    in this case, the `default_criterion` will be used for each task.

    `tasks` can also be a dictionary mapping task names to pytorch loss modules.
    in this case, `default_criterion` is ignored.

    All keys in `tasks` should exist in both `inputs` and `targets`.
    """

    def __init__(
        self,
        tasks: Iterable[str] | dict[str, torch.nn.Module] = ["energy", "forces"],
        weights: Iterable[float] | dict[str, float] = None,
        adaptive_weights: bool = False,
        default_criterion: torch.nn.Module = torch.nn.MSELoss(),
        scale_per_atom: str | Iterable[str] | None = "energy",
        pseudolog_forces: bool = False,
        structurewise_force_loss: bool = False,
    ):
        super().__init__()
        self.pseudolog_forces = pseudolog_forces

        if isinstance(tasks, dict):
            self.tasks = tasks
        else:
            self.tasks = {task: default_criterion for task in tasks}

        if weights is not None and len(weights) != len(tasks):
            raise ValueError(
                f"loss weights {weights} should match number of tasks {tasks.keys()}"
            )

        if isinstance(tasks, dict) and not isinstance(weights, dict):
            warnings.warn(
                "MultitaskLoss: implicitly setting weights based on dictionary order!"
            )

        if weights is None:
            _weights = torch.ones(len(self.tasks))
        elif isinstance(weights, dict):
            _weights = torch.tensor([weights[task] for task in self.tasks.keys()])
        else:
            _weights = torch.tensor(weights)

        self.adaptive_weights = adaptive_weights
        if adaptive_weights:
            # use adaptive task uncertainty from https://arxiv.org/abs/1705.07115
            # model log variance for each task
            self.register_parameter("weights", torch.nn.Parameter(_weights.log()))
        else:
            self.register_buffer("weights", _weights)

        self.scale_per_atom = set()
        if isinstance(scale_per_atom, str):
            self.scale_per_atom.add(scale_per_atom)
        elif isinstance(scale_per_atom, Iterable):
            self.scale_per_atom.update(scale_per_atom)

        # structurewise force loss aggregation from https://openreview.net/forum?id=PfPnugdxup
        self.structurewise_force_loss = structurewise_force_loss
        if self.structurewise_force_loss:
            # modify the force criterion reduction!
            self.tasks["forces"].reduction = "none"

    @property
    def tasknames(self):
        return list(self.tasks.keys())

    def output_transforms(self):
        for task, func in self.tasks.items():
            if func is not None:
                yield task, self.make_output_transform(task)

    def make_output_transform(self, name: str):
        """Build ignite metric transforms for multi-output models.

        `output` should be Tuple[Dict[str,torch.Tensor], Dict[str,torch.Tensor]]
        """

        def output_transform(output):
            """Select output tensor for metric computation."""
            pred, target = output
            pred = pred.get(name)
            target = target.get(name)
            if target is None:
                target = torch.zeros_like(pred)

            if name in ("stress", "virial"):
                pred = einops.rearrange(pred, "b n1 n2 -> b (n1 n2)")
                target = einops.rearrange(target, "b n1 n2 -> b (n1 n2)")

            return pred, target

        return output_transform

    def forward(self, inputs, targets):
        """Accumulate weighted multitask loss

        scale inputs and targets by n_atoms for selected tasks
        """
        n_atoms = targets["n_atoms"]
        if self.weights.data.device != n_atoms.device:
            weights = self.weights.detach().clone().to(n_atoms.device)
        else:
            weights = self.weights

        losses = []
        for task, criterion in self.tasks.items():

            input = inputs.get(task)
            target = targets.get(task)

            if target is None:
                # handle regularization terms included in model output
                task_loss = input
            elif task in self.scale_per_atom:
                task_loss = criterion(input / n_atoms, target / n_atoms)
            elif task == "forces" and self.pseudolog_forces:
                task_loss = criterion(pseudolog10(input), pseudolog10(target))
            else:
                task_loss = criterion(input, target)

            if task == "forces" and self.structurewise_force_loss:
                # perform custom reduction
                # criterion should be torch.nn.MSELoss
                # sqrt makes this a L2 norm reduction
                atomwise_force_loss = task_loss.sum(dim=-1).sqrt()

                device = atomwise_force_loss.device
                index = torch.from_numpy(  # build up structurewise reduction index
                    np.hstack([[i] * n for i, n in enumerate(n_atoms)]),
                ).to(device)
                # first do atom-wise mean reduction for each structure
                task_loss = torch.zeros(len(n_atoms), device=device).scatter_reduce_(
                    0, index, atomwise_force_loss, reduce="mean"
                )
                # then do structure-wise mean reduction
                task_loss = task_loss.mean()

            losses.append(task_loss)

        loss = torch.hstack(losses)

        if self.adaptive_weights:
            # if adaptive, self.weights are log variances
            # factor of 2 assumes Gaussian likelihood...
            task_variance = 2 * weights.exp()
            loss = (loss / task_variance).sum() + torch.log(
                torch.sqrt(task_variance)
            ).sum()
        else:
            loss = torch.sum(loss * weights)

        return loss


class PerAtomLoss(torch.nn.Module):
    """Scaled loss wrapper to compute extrinsic loss function on a per-atom basis.

    `inputs` and `targets` should be `Dict[str, torch.Tensor]`
    `targets` should contain `"n_atoms"`, the batched system size

    criterion = PerAtomLoss(torch.nn.MSELoss(), key="energy")
    """

    def __init__(
        self, criterion: torch.nn.Module = torch.nn.MSELoss(), key: str = "energy"
    ):
        super().__init__()
        self.criterion = criterion
        self.key = key

    def forward(self, inputs, targets):
        # scale loss by system size
        n_atoms = targets["n_atoms"]
        return self.criterion(inputs[self.key] / n_atoms, targets[self.key] / n_atoms)


@dataclass
class Task:
    key: str
    criterion: torch.nn.Module
    preprocess: Literal["per_atom", "tril", "norm"] | Callable = lambda x: x
    weight: float = 1.0
    adaptive: bool = False

    def __post_init__(self):
        if self.preprocess == "tril":

            def tril(x):
                i, j = np.tril_indices(3)
                return x[..., i, j]

            preprocess_func = tril

        elif self.preprocess == "norm":
            preprocess_func = partial(torch.norm, dim=-1)

        elif self.preprocess == "per_atom":
            preprocess_func = lambda x: x

        else:
            preprocess_func = self.preprocess

        self.preprocess_func = preprocess_func

    def transform(
        self, output: tuple[dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = output

        scale = 1
        if self.preprocess == "per_atom":
            scale = targets["n_atoms"]

        inputs = self.preprocess_func(inputs.get(self.key))
        targets = self.preprocess_func(targets.get(self.key))

        inputs = inputs / scale
        if targets is not None:
            targets = targets / scale

        return inputs, targets


class MultitaskLoss(torch.nn.Module):
    """Multitask loss function wrapper for force field training.

    `inputs` and `targets` should be `Dict[str, torch.Tensor]`
    `targets` should contain `"n_atoms"`, the batched system size
    """

    def __init__(self, tasks: dict[str, Task]):
        super().__init__()

        for task in tasks.values():
            if task.adaptive:
                # TODO: check for adaptive tasks and collect their weights
                # into a trainable log-scaled buffer
                raise NotImplementedError("adaptive task weights not yet supported")

        self.tasks = tasks

    def forward(self, inputs, targets):
        loss = torch.tensor(0.0)

        for task_name, task in self.tasks.items():
            input, target = task.transform((inputs, targets))
            loss += task.weight * task.criterion(input, target)

        return loss


def forcefield_metrics(tasks: dict[str, Task]):
    metrics = {
        "mae_energy": ignite.metrics.MeanAbsoluteError(tasks["energy"].transform),
        "mae_forces": ignite.metrics.MeanAbsoluteError(tasks["forces"].transform),
        "mae_virial": ignite.metrics.MeanAbsoluteError(tasks["virial"].transform),
    }

    if "force_norm" in tasks:

        def force_norm_transform(output):
            inputs, targets = output
            inputs = torch.log(torch.norm(inputs["forces"], dim=-1) + 1e-3)
            targets = torch.log(torch.norm(targets["forces"], dim=-1) + 1e-3)
            return inputs, targets

        metrics["mean_abs_log_acc_forces_norm"] = ignite.metrics.MeanAbsoluteError(
            force_norm_transform
        )

    return metrics


def regularization_metrics(tasks: dict[str, Task]):
    metrics = {}
    for task_name, task in tasks.items():
        if isinstance(task.criterion, RegularizationLoss):

            def transform(output):
                inputs, targets = output
                return inputs.get(task.key)

            metrics[task_name] = ignite.metrics.Average(transform)

    return metrics
