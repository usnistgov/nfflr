from __future__ import annotations

__all__ = ()
import os
from pathlib import Path
from datetime import timedelta
from typing import TYPE_CHECKING

import torch
from torch.utils.data import SubsetRandomSampler

import typer
import ignite
import ignite.distributed as idist
from ignite.utils import manual_seed
from ignite.metrics import Loss, MeanAbsoluteError
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import Checkpoint, DiskSaver, FastaiLRFinder, TerminateOnNan
from ignite.engine import Events, create_supervised_trainer

import ray
import ray.tune
from ray.air import session

from py_config_runner import ConfigObject

import nfflr


if TYPE_CHECKING:
    from nfflr.train.config import TrainingConfig

from nfflr.train.utils import (
    group_decay,
    select_target,
    setup_evaluator_with_grad,
    setup_optimizer,
    setup_scheduler,
    transfer_outputs,
)
from nfflr.train.swag import SWAGHandler
from nfflr.models.utils import reset_initial_output_bias

cli = typer.Typer()

# set up multi-GPU training, if available
gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
num_gpus = len(gpus.split(","))

backend = None
nproc_per_node = None
if num_gpus > 1:
    backend == "nccl" if torch.distributed.is_nccl_available() else "gloo"
    nproc_per_node = num_gpus

spawn_kwargs = {
    "backend": backend,
    "nproc_per_node": nproc_per_node,
    "timeout": timedelta(seconds=60),
}


def log_console(engine: ignite.engine.Engine, name: str):
    """Log evaluation stats to console."""
    epoch = engine.state.training_epoch  # custom state field
    m = engine.state.metrics
    loss = m["loss"]

    print(f"{name} results - Epoch: {epoch}  Avg loss: {loss:.2f}")
    if "mae_forces" in m.keys():
        print(f"energy: {m['mae_energy']:.2f}  force: {m['mae_forces']:.4f}")


def get_dataflow(dataset, config: TrainingConfig):
    """Configure training and validation datasets."""

    train_loader = idist.auto_dataloader(
        dataset,
        collate_fn=dataset.collate,
        batch_size=config.batch_size,
        sampler=SubsetRandomSampler(dataset.split["train"]),
        drop_last=True,
        num_workers=config.dataloader_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = idist.auto_dataloader(
        dataset,
        collate_fn=dataset.collate,
        batch_size=config.batch_size,
        sampler=SubsetRandomSampler(dataset.split["val"]),
        drop_last=False,  # True -> possible issue crashing with MP dataset
        num_workers=config.dataloader_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader


def _initialize(model, criterion, train_loader, config: TrainingConfig):
    """Initialize model, criterion, and optimizer."""
    model = idist.auto_model(model)
    criterion = idist.auto_model(criterion)

    params = group_decay(model)
    if isinstance(criterion, torch.nn.Module) and len(list(criterion.parameters())) > 0:
        params.append({"params": criterion.parameters(), "weight_decay": 0})

    optimizer = setup_optimizer(params, config)
    optimizer = idist.auto_optim(optimizer)

    if model.config.initialize_bias:
        reset_initial_output_bias(
            model, train_loader, max_samples=500 / config.batch_size
        )

    return model, criterion, optimizer


def setup_trainer(
    model, criterion, optimizer, scheduler, prepare_batch, config: TrainingConfig
):
    """Create ignite trainer and attach common event handlers."""
    device = idist.device()

    trainer = create_supervised_trainer(
        model,
        optimizer,
        criterion,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        prepare_batch=prepare_batch,
        device=device,
    )

    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    if scheduler is not None:
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
        )

    return trainer


def setup_checkpointing(state: dict, config: TrainingConfig):
    """Configure model and trainer checkpointing.

    `state` should contain at least `model`, `optimizer`, and `trainer`.
    """

    checkpoint_handler = Checkpoint(
        state,
        DiskSaver(config.output_dir, create_dir=True, require_empty=False),
        n_saved=1,
        global_step_transform=lambda *_: state["trainer"].state.epoch,
    )
    state["trainer"].add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)

    if config.resume_checkpoint is not None:
        checkpoint = torch.load(config.resume_checkpoint, map_location=idist.device())
        Checkpoint.load_objects(to_load=state, checkpoint=checkpoint)

    return state


def setup_evaluators(model, prepare_batch, metrics, transfer_outputs):
    """Configure train and validation evaluators."""
    device = idist.device()
    # create_supervised_evaluator
    train_evaluator = setup_evaluator_with_grad(
        model,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
        output_transform=transfer_outputs,
    )

    val_evaluator = setup_evaluator_with_grad(
        model,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
        output_transform=transfer_outputs,
    )

    return train_evaluator, val_evaluator


def run_train(local_rank: int, config):
    """Nfflr trainer entry point."""
    model = config["model"]
    criterion = config["criterion"]
    dataset = config["dataset"]
    config = config["trainer"]

    rank = idist.get_rank()
    manual_seed(config.random_seed + local_rank)

    train_loader, val_loader = get_dataflow(dataset, config)
    model, criterion, optimizer = _initialize(model, criterion, train_loader, config)
    scheduler = setup_scheduler(config, optimizer, len(train_loader))

    if config.swag:
        swag_handler = SWAGHandler(model)

    trainer = setup_trainer(
        model, criterion, optimizer, scheduler, dataset.prepare_batch, config
    )
    if config.progress and rank == 0:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    if config.checkpoint:
        state = dict(model=model, optimizer=optimizer, trainer=trainer)

        if scheduler is not None:
            state["scheduler"] = scheduler
        if isinstance(criterion, torch.nn.Module):
            state["criterion"] = criterion
        if config.swag:
            state["swagmodel"] = swag_handler.swagmodel

        setup_checkpointing(state, config)

    if config.swag:
        swag_handler.attach(trainer)

    # evaluation
    metrics = {"loss": Loss(criterion)}
    if isinstance(criterion, nfflr.nn.MultitaskLoss):
        # NOTE: unscaling currently uses a global scale
        # shared across all tasks (intended to scale energy units)
        unscale = None
        if dataset.standardize:
            unscale = dataset.scaler.unscale

        eval_metrics = {
            f"mae_{task}": MeanAbsoluteError(select_target(task, unscale_fn=unscale))
            for task in criterion.tasks
        }
        metrics.update(eval_metrics)

    train_evaluator, val_evaluator = setup_evaluators(
        model, dataset.prepare_batch, metrics, transfer_outputs
    )
    if config.progress and rank == 0:
        vpbar = ProgressBar()
        vpbar.attach(train_evaluator)
        vpbar.attach(val_evaluator)

    history = {
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
    }

    @trainer.on(Events.EPOCH_COMPLETED)
    def _eval(engine):
        n_train_eval = int(config.train_eval_fraction * len(train_loader))
        n_train_eval = max(n_train_eval, 1)  # at least one batch
        train_evaluator.state.training_epoch = engine.state.epoch
        val_evaluator.state.training_epoch = engine.state.epoch
        train_evaluator.run(train_loader, epoch_length=n_train_eval, max_epochs=1)
        val_evaluator.run(val_loader)

    def log_metric_history(engine, output_dir: Path):
        phases = {"train": train_evaluator, "validation": val_evaluator}
        for phase, evaluator in phases.items():
            for key, value in evaluator.state.metrics.items():
                history[phase][key].append(value)
        torch.save(history, output_dir / "metric_history.pkl")

    if rank == 0:
        train_evaluator.add_event_handler(Events.COMPLETED, log_console, name="train")
        val_evaluator.add_event_handler(Events.COMPLETED, log_console, name="val")
        val_evaluator.add_event_handler(
            Events.COMPLETED, log_metric_history, config.output_dir
        )

        if ray.tune.is_session_enabled():
            val_evaluator.add_event_handler(
                Events.COMPLETED, lambda engine: session.report(engine.state.metrics)
            )

    trainer.run(train_loader, max_epochs=config.epochs)

    return val_evaluator.state.metrics["loss"]


def run_lr(local_rank: int, config):
    config["checkpoint"] = False
    scheduler = None

    model = config["model"]
    criterion = config["criterion"]
    dataset = config["dataset"]
    config = config["trainer"]

    rank = idist.get_rank()
    manual_seed(config.random_seed + local_rank)

    train_loader, val_loader = get_dataflow(dataset, config)
    model, criterion, optimizer = _initialize(model, criterion, train_loader, config)
    trainer = setup_trainer(
        model, criterion, optimizer, scheduler, dataset.prepare_batch, config
    )

    lr_finder = FastaiLRFinder()
    to_save = {"model": model, "optimizer": optimizer}
    with lr_finder.attach(
        trainer, to_save, start_lr=1e-6, end_lr=0.1, num_iter=400, diverge_th=1e9
    ) as finder:
        finder.run(train_loader)

    if rank == 0:
        # print("Suggested LR", lr_finder.lr_suggestion())
        ax = lr_finder.plot(display_suggestion=False)
        ax.loglog()
        ax.set_ylim(None, 5.0)
        ax.figure.savefig("lr.png")


@cli.command()
def train(config_path: Path, verbose: bool = False):
    """NFF training entry point."""
    with idist.Parallel(**spawn_kwargs) as parallel:
        config = ConfigObject(config_path)
        if verbose:
            print(config)
        parallel.run(run_train, config)


@cli.command()
def lr(config_path: Path, verbose: bool = False):
    """NFF Learning rate finder entry point."""
    with idist.Parallel(**spawn_kwargs) as parallel:
        if verbose:
            print(spawn_kwargs)
            print("loading config")

        config = ConfigObject(config_path)

        if verbose:
            print(config)

        parallel.run(run_lr, config)


if __name__ == "__main__":
    cli()
