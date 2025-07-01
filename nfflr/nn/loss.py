import warnings
from collections.abc import Iterable
from typing import Sequence

import torch
import einops
import numpy as np
import ignite.metrics
from ignite.metrics import MeanAbsoluteError

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
        print(f"{inputs=}")
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

    def metrics(self):
        metrics = {
            f"mae_{task}": MeanAbsoluteError(transform)
            for task, transform in self.output_transforms()
        }

        # metrics = {
        #     f"med_abs_err_{task}": MedianAbsoluteError(transform)
        #     for task, transform in self.output_transforms()
        # }
        # if idist.get_world_size() == 1:
        #     metrics.update(eval_metrics)
        # else:
        #     warnings.warn(
        #         "MedianAbsoluteError metric not yet supported in distributed training"
        #     )

        return metrics

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


class Norm(torch.nn.Module):
    def forward(self, x):
        return x.norm(dim=-1)


class AddConstant(torch.nn.Module):
    def __init__(self, value: float = 1e-3):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x + self.value


class FlattenTril(torch.nn.Module):
    def __init__(self, size: int = 3):
        super().__init__()
        self.size = size
        i, j = np.tril_indices(size)
        self.i = i
        self.j = j

    def forward(self, x):
        return x[..., self.i, self.j]


class ScaledAsinh(torch.nn.Module):
    def __init__(self, scale: float = 1e-3):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return torch.asinh(x / self.scale)


class Task(torch.nn.Module):
    """Force field task definition.

    ```python
    tasks = {
        "energy": Task("energy", scale_per_atom=True),
        "force_norm": Task("forces", transform=Norm())
    }
    criterion = MultitaskLoss(tasks)
    metrics = {
        f"mae_{key}": ignite.metrics.MeanAbsoluteError(tasks[key].process_outputs)
        for key in ("energy", "forces", "virial")
    }
    ```
    """

    def __init__(
        self,
        key: str,
        weight: float = 1.0,
        adaptive: bool = False,
        criterion: torch.nn.Module = torch.nn.MSELoss(),
        scale_per_atom: bool = False,
        transform: torch.nn.Module | Sequence[torch.nn.Module] = torch.nn.Identity(),
    ):
        super().__init__()
        self.key = key
        self.weight = weight
        self.adaptive = adaptive
        self.criterion = criterion
        self.scale_per_atom = scale_per_atom

        if isinstance(transform, Sequence):
            self.transform = torch.nn.Sequential(*transform)
        else:
            self.transform = transform

    def forward(self, x: tuple[dict[str, torch.Tensor]]):
        inputs, targets = self.process_outputs(x)
        return self.criterion(inputs, targets)

    def process_outputs(self, x: tuple[dict[str, torch.Tensor]]):
        # should `forward` process outputs, or call the criterion?
        inputs, targets = x

        scale = 1
        if self.scale_per_atom:
            # apply scaling before transform
            scale = targets["n_atoms"]

        inputs = self.transform(inputs[self.key] / scale)

        if self.key in targets:
            targets = self.transform(targets[self.key] / scale)
        else:
            targets = None

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

    def forward(self, inputs, targets, reduce_terms: bool = True):

        losses = {}
        for task_name, task in self.tasks.items():
            losses[task_name] = task.weight * task((inputs, targets))

        if not reduce_terms:
            return losses

        return sum(losses.values())


def forcefield_metrics(tasks: dict[str, Task]):
    metrics = {
        "mae_energy": ignite.metrics.MeanAbsoluteError(tasks["energy"].process_outputs),
        "mae_forces": ignite.metrics.MeanAbsoluteError(tasks["forces"].process_outputs),
        "mae_virial": ignite.metrics.MeanAbsoluteError(tasks["virial"].process_outputs),
    }

    metrics["mae_force_norm"] = ignite.metrics.MeanAbsoluteError(
        output_transform=Task("forces", transform=Norm()).process_outputs
    )
    metrics["mae_relative_force_norm"] = ignite.metrics.MeanAbsoluteError(
        output_transform=Task(
            "forces", transform=(Norm(), ScaledAsinh(1e-3))
        ).process_outputs
    )

    return metrics


def regularization_metrics(tasks: dict[str, Task]):
    metrics = {}
    for task_name, task in tasks.items():
        if isinstance(task.criterion, RegularizationLoss):
            metrics[task_name] = ignite.metrics.Average(task)

    return metrics
