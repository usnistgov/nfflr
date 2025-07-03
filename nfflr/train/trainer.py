from __future__ import annotations

__all__ = ()
import os
from pathlib import Path
from datetime import timedelta
from typing import TYPE_CHECKING
import warnings

import torch
from torch.utils.data import SubsetRandomSampler

import typer
import ignite
import ignite.distributed as idist
from ignite.utils import manual_seed
from ignite.metrics import Loss
from ignite.contrib.metrics import GpuInfo
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
    setup_evaluator_with_grad,
    setup_optimizer,
    setup_scheduler,
    transfer_outputs,
    reset_initial_output_bias,
)
from nfflr.train.swag import SWAGHandler

cli = typer.Typer()

# set up multi-GPU training, if available
gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
if gpus:
    num_gpus = len(gpus.split(","))
else:
    num_gpus = 0

backend = None
nproc_per_node = None
if num_gpus > 1:
    backend = "nccl" if torch.distributed.is_nccl_available() else "gloo"
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
    if "med_abs_err_forces" in m.keys():
        err_energy = m["med_abs_err_energy"]
        err_forces = m["med_abs_err_forces"]
        print(f"median abs err: energy: {err_energy:.2f}  force: {err_forces:.4f}")
    elif "mae_forces" in m.keys():
        print(f"energy: {m['mae_energy']:.2f}  force: {m['mae_forces']:.4f}")


def get_dataflow(dataset: nfflr.AtomsDataset, config: TrainingConfig):
    """Configure training and validation datasets.

    Wraps `train` and `val` splits of `dataset` in ignite's
    :py:func:`auto_dataloader <ignite.distributed.auto.auto_dataloader>`.


    Parameters
    ----------
    dataset : nfflr.AtomsDataset
    config : nfflr.train.TrainingConfig
    """
    batch_size = config.per_device_batch_size * idist.get_world_size()
    train_loader = idist.auto_dataloader(
        dataset,
        collate_fn=dataset.collate,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(dataset.split["train"]),
        drop_last=True,
        num_workers=config.dataloader_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = idist.auto_dataloader(
        dataset,
        collate_fn=dataset.collate,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(dataset.split["val"]),
        drop_last=False,  # True -> possible issue crashing with MP dataset
        num_workers=config.dataloader_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader


def setup_model_and_optimizer(
    model: torch.nn.Module,
    dataset: nfflr.AtomsDataset,
    config: TrainingConfig,
):
    """Initialize model, criterion, and optimizer."""
    model = idist.auto_model(model)

    criterion = config.criterion
    if isinstance(criterion, torch.nn.Module) and any(
        [p.requires_grad for p in criterion.parameters()]
    ):
        criterion = idist.auto_model(criterion)

    if isinstance(criterion, nfflr.nn.MultitaskLoss):
        # auto_model won't transfer buffers...?
        criterion = criterion.to(idist.device())

    params = group_decay(model)
    if isinstance(criterion, torch.nn.Module) and len(list(criterion.parameters())) > 0:
        params.append({"params": criterion.parameters(), "weight_decay": 0})

    optimizer = setup_optimizer(params, config)
    optimizer = idist.auto_optim(optimizer)

    if config.initialize_estimated_reference_energies:
        model.reset_atomic_reference_energies(dataset.estimate_reference_energies())

    if config.initialize_bias:
        train_loader, _ = get_dataflow(dataset, config)
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
    if torch.cuda.is_available():
        GpuInfo().attach(trainer, name="gpu")

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
        # if the model is wrapped in SWAGHandler, model requires surgery
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


def train(
    model: torch.nn.Module,
    dataset: nfflr.AtomsDataset,
    config: nfflr.train.TrainingConfig,
    local_rank: int = 0,
):
    """NFFLr trainer entry point.

    Parameters
    ----------
    model : torch.nn.Module
    dataset : nfflr.AtomsDataset
    config : nfflr.train.TrainingConfig
    local_rank : int, optional
    """
    rank = idist.get_rank()
    manual_seed(config.random_seed + local_rank)

    train_loader, val_loader = get_dataflow(dataset, config)
    model, criterion, optimizer = setup_model_and_optimizer(model, dataset, config)
    scheduler = setup_scheduler(config, optimizer, len(train_loader))

    if config.swag_start is not None:
        swag_handler = SWAGHandler(model)
        if config.resume_checkpoint is not None:
            # have to manually load state to reset parameter buffer sizes
            swag_handler.swagmodel.load_state(config.resume_checkpoint)

    trainer = setup_trainer(
        model, criterion, optimizer, scheduler, dataset.prepare_batch, config
    )
    if config.progress and rank == 0:
        pbar = ProgressBar()
        pbar.attach(trainer, metric_names=["gpu:0 mem(%)", "gpu:0 util(%)"])

    if config.checkpoint:
        state = dict(model=model, optimizer=optimizer, trainer=trainer)

        if scheduler is not None:
            state["scheduler"] = scheduler
        if isinstance(criterion, torch.nn.Module):
            state["criterion"] = criterion
        if config.swag_start is not None:
            state["swagmodel"] = swag_handler.swagmodel

        setup_checkpointing(state, config)

    if config.swag_start is not None:
        swag_handler.attach(
            trainer, event=ignite.engine.Events.EPOCH_COMPLETED(after=config.swag_start)
        )

    # evaluation
    metrics = {"loss": Loss(criterion)}
    if config.metrics is not None:
        metrics.update(config.metrics)

    history = {
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
        "lr": [],
    }
    if config.resume_checkpoint is not None:
        history = torch.load(config.output_dir / "metric_history.pkl")

    train_evaluator, val_evaluator = setup_evaluators(
        model, dataset.prepare_batch, metrics, transfer_outputs
    )
    if config.progress and rank == 0:
        vpbar = ProgressBar()
        vpbar.attach(train_evaluator)
        vpbar.attach(val_evaluator)

    @trainer.on(Events.EPOCH_COMPLETED)
    def _eval(engine):
        n_train_eval = int(config.train_eval_fraction * len(train_loader))
        n_train_eval = max(n_train_eval, 1)  # at least one batch
        train_evaluator.state.training_epoch = engine.state.epoch
        val_evaluator.state.training_epoch = engine.state.epoch
        train_evaluator.run(train_loader, epoch_length=n_train_eval, max_epochs=1)
        val_evaluator.run(val_loader)

    @trainer.on(Events.ITERATION_COMPLETED)
    def _log_lr(engine):
        history["lr"].append(scheduler.get_last_lr()[0])

    def _log_task_weights(engine):
        log_variance = criterion.log_variance.detach().tolist()
        if "log_task_variance" not in history:
            history["log_task_variance"] = [log_variance]
        else:
            history["log_task_variance"].append(log_variance)

        return

    def log_metric_history(engine, output_dir: Path):
        phases = {"train": train_evaluator, "validation": val_evaluator}
        for phase, evaluator in phases.items():
            for key, value in evaluator.state.metrics.items():
                history[phase][key].append(value)
        torch.save(history, output_dir / "metric_history.pkl")
        return

    if rank == 0:
        train_evaluator.add_event_handler(Events.COMPLETED, log_console, name="train")
        val_evaluator.add_event_handler(Events.COMPLETED, log_console, name="val")
        if isinstance(criterion, nfflr.nn.MultitaskLoss) and criterion.adaptive:
            val_evaluator.add_event_handler(
                Events.COMPLETED, _log_task_weights, config.output_dir
            )

        val_evaluator.add_event_handler(
            Events.COMPLETED, log_metric_history, config.output_dir
        )

        if ray.train._internal.session.get_session():
            val_evaluator.add_event_handler(
                Events.COMPLETED, lambda engine: session.report(engine.state.metrics)
            )

    max_epochs = config.epochs
    if config.swalr_epochs is not None:
        max_epochs = config.epochs + config.swalr_epochs

    trainer.run(train_loader, max_epochs=max_epochs)

    return val_evaluator.state.metrics["loss"]


def lr(
    model: torch.nn.Module,
    dataset: nfflr.AtomsDataset,
    config: nfflr.train.TrainingConfig,
    local_rank: int = 0,
):
    """NFFLr learning rate finder entry point.

    Runs the Fast.ai learning rate finder
    :py:class:`ignite.handlers.lr_finder.FastaiLRFinder`
    for `model`, `dataset`, and `config`.

    Parameters
    ----------
    model : torch.nn.Module
    dataset : nfflr.AtomsDataset
    config : nfflr.train.TrainingConfig
    local_rank : int, optional
    """
    rank = idist.get_rank()
    manual_seed(config.random_seed + local_rank)
    config.checkpoint = False
    scheduler = None

    train_loader, val_loader = get_dataflow(dataset, config)
    model, criterion, optimizer = setup_model_and_optimizer(model, dataset, config)
    trainer = setup_trainer(
        model, criterion, optimizer, scheduler, dataset.prepare_batch, config
    )

    if config.progress and rank == 0:
        pbar = ProgressBar()
        pbar.attach(trainer, metric_names=["gpu:0 mem(%)", "gpu:0 util(%)"])

    lr_finder = FastaiLRFinder()
    to_save = {"model": model, "optimizer": optimizer}
    with lr_finder.attach(trainer, to_save, num_iter=400, diverge_th=1e9) as finder:
        finder.run(train_loader)

    if rank == 0:
        import matplotlib.pyplot as plt

        # print("Suggested LR", lr_finder.lr_suggestion())
        lrs = lr_finder._history["lr"]
        losses = lr_finder._history["loss"]
        print(f"{losses=}")
        plt.semilogx(lrs, losses)
        plt.xlabel("lr")
        plt.ylabel("loss")
        plt.savefig("lr.png")
        torch.save(dict(lr=lrs, losses=losses), "lr.pkl")
        # ax = lr_finder.plot(display_suggestion=False)
        # ax.loglog()
        # ax.set_ylim(None, 5.0)
        # ax.figure.savefig("lr.png")


def train_wrapper(local_rank, model, dataset, args):
    """Wrap train entry point for idist.Parallel."""
    return train(model, dataset, args, local_rank=local_rank)


@cli.command("train")
def cli_train(config_path: Path, verbose: bool = False):
    """NFF training entry point."""
    with idist.Parallel(**spawn_kwargs) as parallel:
        config = ConfigObject(config_path)
        if verbose:
            print(config)
            print(spawn_kwargs)
            print("loading config")

        parallel.run(train_wrapper, config.model, config.dataset, config.args)


def lr_wrapper(local_rank, model, dataset, args):
    """Wrap lr entry point for idist.Parallel."""
    return lr(model, dataset, args, local_rank=local_rank)


@cli.command("lr")
def cli_lr(config_path: Path, verbose: bool = False):
    """NFF Learning rate finder entry point."""
    with idist.Parallel(**spawn_kwargs) as parallel:
        if verbose:
            print(spawn_kwargs)
            print("loading config")

        config = ConfigObject(config_path)

        if verbose:
            print(config)

        parallel.run(lr_wrapper, config.model, config.dataset, config.args)


if __name__ == "__main__":
    cli()
