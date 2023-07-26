"""Common training setup utility functions."""
from typing import Any, Callable, Dict, Tuple

import torch
from torch import nn
from ignite.engine import Engine
from ignite.engine import Events, create_supervised_trainer


def multitarget_loss(criteria: Dict[dict, nn.Module], scales: Dict[str, float]):
    def loss(outputs, targets):
        return sum(
            scales.get(key, 1.0) * criteria[key](outputs[key], targets[key])
            for key in criteria
        )

    return loss


def losswrapper(criterion):
    def loss(outputs, targets):
        return sum(criterion(outputs[key], targets[key]) for key in outputs)

    return loss


def setup_criterion(config):
    """Optimization criterion helper function.

    config["criterion"] can be

    - a pytorch loss (nn.Module)
    - dict[str, nn.Module]
    - a `Callable`
    """
    criterion = config["criterion"]
    forcefield = config["dataset"].target == "energy_and_forces"

    if forcefield:
        if isinstance(criterion, dict):
            loss_scale = config.get("loss_scale", {})
            return multitarget_loss(criterion, loss_scale)
        if isinstance(criterion, nn.Module):
            return losswrapper(criterion)

    return criterion


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


def select_target(name: str):
    """Build ignite metric transforms for multi-output models.

    `output` should be Tuple[Dict[str,torch.Tensor], Dict[str,torch.Tensor]]
    """

    def output_transform(output):
        """Select output tensor for metric computation."""
        pred, target = output
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


def setup_optimizer(params, config):
    """Set up optimizer for param groups."""
    if isinstance(config["optimizer"], torch.optim.Optimizer):
        optimizer = config["optimizer"]
    if config["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config["learning_rate"],
            momentum=0.9,
            weight_decay=config["weight_decay"],
        )
    return optimizer


def setup_scheduler(config, optimizer, steps_per_epoch: int | float):
    """Configure OneCycle scheduler."""
    warmup_steps = config.get("warmup_steps", 0.3)
    if warmup_steps < 1:
        # fractional specification
        pct_start = warmup_steps
    else:
        pct_start = config["warmup_steps"] / (config["epochs"] * steps_per_epoch)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        epochs=config["epochs"],
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
    )

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
