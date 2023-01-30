"""Common training setup utility functions."""
from typing import Any, Callable

import torch
from ignite.engine import Engine

from alignn.config import TrainingConfig


def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def setup_optimizer(params, config: TrainingConfig):
    """Set up optimizer for param groups."""
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


def setup_evaluator_with_grad(
    model,
    metrics,
    prepare_batch,
    device="cpu",
    non_blocking=False,
    output_transform: Callable[[Any, Any, Any], Any] = lambda x, y, y_pred: (
        y_pred,
        y,
    ),
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
