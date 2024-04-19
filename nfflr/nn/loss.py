import warnings
from collections.abc import Iterable

import torch

from nfflr.train.evaluation import pseudolog10

class MultitaskLoss(torch.nn.Module):
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

    @property
    def tasknames(self):
        return list(self.tasks.keys())

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
            input, target = inputs[task], targets[task]

            if task in self.scale_per_atom:
                losses.append(criterion(input / n_atoms, target / n_atoms))
            elif task == "forces" and self.pseudolog_forces:
                losses.append(criterion(pseudolog10(input), pseudolog10(target)))
            else:
                losses.append(criterion(input, target))

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
