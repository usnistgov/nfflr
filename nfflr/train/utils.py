"""Common training setup utility functions."""
from __future__ import annotations
from typing import Any, Callable, TYPE_CHECKING
from pathlib import Path

import torch
from torch import nn
from ignite.engine import Engine
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from nfflr.train.config import TrainingConfig


def transfer_outputs(x, y, y_pred):
    """Convert outputs for evaluators / metric computation.

    return the format expected by loss function - predict first.
    """
    if isinstance(y_pred, dict):
        return tuple(
            {key: value.detach().cpu() for key, value in xs.items()}
            for xs in (y_pred, y)
        )

    return y_pred.detach().cpu(), y.detach().cpu()


def transfer_outputs_eos(outputs):
    """Save outputs.

    list of outputs for each batch
    in this case List[Tuple[Dict[str,torch.Tensor], Dict[str,torch.Tensor]]]
    """
    y_pred, y = outputs

    if isinstance(y_pred, dict):
        return tuple(
            {key: value.detach().cpu() for key, value in xs.items()}
            for xs in (y_pred, y)
        )

    return y_pred.detach().cpu(), y.detach().cpu()


def select_target(name: str, unscale_fn: Callable = None):
    """Build ignite metric transforms for multi-output models.

    `output` should be Tuple[Dict[str,torch.Tensor], Dict[str,torch.Tensor]]
    """

    def output_transform(output):
        """Select output tensor for metric computation."""
        pred, target = output
        if unscale_fn is not None:
            return unscale_fn(pred[name]), unscale_fn(target[name])
        return pred[name], target[name]

    return output_transform


def default_select_decay(model):
    decay, no_decay = [], []

    for mname, m in model.named_modules():
        for name, p in m.named_parameters(recurse=False):
            if isinstance(m, nn.Linear) and name == "weight" and not mname == "fc":
                decay.append(p)
            else:
                no_decay.append(p)

    return decay, no_decay


def named_select_decay(model, exclude_biases=True):
    no_decay_names = model.no_weight_decay()

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if name in no_decay_names or "bias" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return decay, no_decay


def group_decay(model):
    """Omit weight decay from everything but `Linear` weights."""
    if hasattr(model, "no_weight_decay"):
        decay, no_decay = named_select_decay(model)
    else:
        decay, no_decay = default_select_decay(model)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def setup_optimizer(params, config: TrainingConfig):
    """Set up optimizer for param groups."""
    if isinstance(config.optimizer, torch.optim.Optimizer):
        optimizer = config.optimizer
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    return optimizer


def setup_scheduler(config: TrainingConfig, optimizer, steps_per_epoch: int | float):
    """Configure OneCycle scheduler."""

    warmup_steps = config.warmup_steps
    if warmup_steps < 1:
        # fractional specification
        pct_start = warmup_steps
    else:
        pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)

    if config.epochs == 0:
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
        )

    if config.swag_epochs is not None:

        # TODO: SWALR expects epochs, OneCycle expects steps...
        swa_start = config.epochs
        swalr = torch.optim.swa_utils.SWALR(
            optimizer,
            anneal_epochs=config.swag_anneal_epochs,
            swa_lr=config.swag_learning_rate,
        )
        if scheduler is not None:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [scheduler, swalr], milestones=[swa_start]
            )
        else:
            scheduler = swalr

    return scheduler


def default_transfer_outputs(x, y, y_pred):
    """Default evaluation output transformation.

    Return y_pred and y in the order expected by metrics
    """
    return (y_pred, y)


def setup_evaluator_with_grad(
    model,
    metrics,
    prepare_batch,
    device="cpu",
    non_blocking=False,
    output_transform: Callable[[Any, Any, Any], Any] = default_transfer_outputs,
):
    """Set up custom ignite evaluation engine.

    Closely follows
    https://github.com/pytorch/ignite/blob/master/ignite/engine/__init__.py#L558
    with the exception of keeping gradient computation
    """

    def evaluation_step(engine, batch):
        """Evaluate step, including gradient tracing."""
        model.eval()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)

        # make sure to zero gradients in eval step?
        for param in model.parameters():
            param.grad = None

        return output_transform(x, y, y_pred)

    evaluator = Engine(evaluation_step)
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def parity_plots(engine: Engine, directory: str | Path, name: str = "train"):
    """Plot predictions for energy and forces.

    # attach to evaluators:
    eos_train = EpochOutputStore(output_transform=transfer_outputs_eos)
    eos_train.attach(train_evaluator, "output")
    eos_val = EpochOutputStore(output_transform=transfer_outputs_eos)
    eos_val.attach(val_evaluator, "output")

    """
    epoch = engine.state.training_epoch  # custom state field
    output = engine.state.output

    fig, axes = plt.subplots(ncols=2, figsize=(16, 8))

    for pred, tgt in output:
        axes[0].scatter(
            tgt["energy"].cpu().detach().numpy(),
            pred["energy"].cpu().detach().numpy(),
            # color="k",
        )
        axes[0].set(xlabel="DFT energy", ylabel="predicted energy")
        axes[1].scatter(
            tgt["forces"].cpu().detach().numpy(),
            pred["forces"].cpu().detach().numpy(),
            # color="k",
        )
        axes[1].set(xlabel="DFT force", ylabel="predicted force")
        axes[1].set_ylim(-10, 10)

    plt.tight_layout()
    plt.savefig(Path(directory) / f"parity_plots_{name}_{epoch:03d}.png")
    plt.clf()
    plt.close()
