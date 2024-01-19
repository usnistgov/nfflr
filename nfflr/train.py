__all__ = ()
import os
from pathlib import Path
from datetime import timedelta

import torch
from torch.utils.data import SubsetRandomSampler

import typer
import matplotlib.pyplot as plt
import ignite.distributed as idist
from ignite.utils import manual_seed
from ignite.metrics import Loss, MeanAbsoluteError
from ignite.handlers.stores import EpochOutputStore
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import Checkpoint, DiskSaver, FastaiLRFinder, TerminateOnNan
from ignite.engine import Events, create_supervised_trainer

import ray
from ray.air import session

from py_config_runner import ConfigObject

from nfflr.training_utils import (
    group_decay,
    select_target,
    setup_evaluator_with_grad,
    setup_optimizer,
    setup_scheduler,
    transfer_outputs,
    transfer_outputs_eos,
    setup_criterion,
)

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


def log_results(engine, name):
    epoch = engine.state.training_epoch  # custom state field
    m = engine.state.metrics
    loss = m["loss"]

    print(f"{name} results - Epoch: {epoch}  Avg loss: {loss:.2f}")
    if "mae_forces" in m.keys():
        print(f"energy: {m['mae_energy']:.2f}  force: {m['mae_forces']:.4f}")

    # for key, value in m.items():
    #     history[name][key].append(value)


def parity_plots(engine, directory, name="train"):
    """Plot predictions for energy and forces."""
    epoch = engine.state.training_epoch  # custom state field
    output = engine.state.output
    # output = getattr(engine.state, f"output_{name}")

    fig, axes = plt.subplots(ncols=2, figsize=(16, 8))

    for (pred, tgt) in output:

        axes[0].scatter(
            tgt["total_energy"].cpu().detach().numpy(),
            pred["total_energy"].cpu().detach().numpy(),
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


def get_dataflow(config):
    # configure training and validation datasets...

    dataset = config["dataset"]

    train_loader = idist.auto_dataloader(
        dataset,
        collate_fn=dataset.collate,
        batch_size=config["batch_size"],
        sampler=SubsetRandomSampler(dataset.split["train"]),
        drop_last=True,
        num_workers=config["num_workers"],
        pin_memory="cuda" in idist.device().type,
    )
    val_loader = idist.auto_dataloader(
        dataset,
        collate_fn=dataset.collate,
        batch_size=config["batch_size"],
        sampler=SubsetRandomSampler(dataset.split["val"]),
        drop_last=True,  # needed to prevent crashing with MP dataset
        num_workers=config["num_workers"],
        pin_memory="cuda" in idist.device().type,
    )

    return train_loader, val_loader


def _initialize(config, steps_per_epoch):
    model = idist.auto_model(config["model"])
    params = group_decay(model)
    optimizer = setup_optimizer(params, config)
    optimizer = idist.auto_optim(optimizer)
    scheduler = setup_scheduler(config, optimizer, steps_per_epoch)

    return model, optimizer, scheduler


def setup_trainer(rank, model, optimizer, scheduler, config):
    device = idist.device()

    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    trainer = create_supervised_trainer(
        model,
        optimizer,
        setup_criterion(config),
        gradient_accumulation_steps=gradient_accumulation_steps,
        prepare_batch=config["dataset"].prepare_batch,
        device=device,
    )

    if config.get("progress", True) and rank == 0:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    if scheduler is not None:
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
        )

    if not config.get("checkpoint", True):
        return trainer

    to_save = dict(
        model=model,
        optimizer=optimizer,
        trainer=trainer,
    )

    if scheduler is not None:
        to_save["scheduler"] = scheduler

    checkpoint_handler = Checkpoint(
        to_save,
        DiskSaver(config["output_dir"], create_dir=True, require_empty=False),
        n_saved=1,
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    if rank == 0:
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)

    return trainer


def setup_evaluators(rank, model, config, metrics, transfer_outputs):
    device = idist.device()
    # create_supervised_evaluator
    train_evaluator = setup_evaluator_with_grad(
        model,
        metrics=metrics,
        prepare_batch=config["dataset"].prepare_batch,
        device=device,
        output_transform=transfer_outputs,
    )

    val_evaluator = setup_evaluator_with_grad(
        model,
        metrics=metrics,
        prepare_batch=config["dataset"].prepare_batch,
        device=device,
        output_transform=transfer_outputs,
    )

    if config["progress"] and rank == 0:
        vpbar = ProgressBar()
        vpbar.attach(train_evaluator)
        vpbar.attach(val_evaluator)

    return train_evaluator, val_evaluator


def run_train(local_rank: int, config):
    rank = idist.get_rank()
    manual_seed(config["random_seed"] + local_rank)

    train_loader, val_loader = get_dataflow(config)

    # TODO: add criterion to this...
    model, optimizer, scheduler = _initialize(config, len(train_loader))
    trainer = setup_trainer(rank, model, optimizer, scheduler, config)

    # evaluation
    metrics = config.get("metrics", {})
    metrics.update({"loss": Loss(setup_criterion(config))})

    if config["dataset"].target == "energy_and_forces":
        metrics.update(
            {
                "mae_energy": MeanAbsoluteError(select_target("total_energy")),
                "mae_forces": MeanAbsoluteError(select_target("forces")),
            }
        )

    train_evaluator, val_evaluator = setup_evaluators(
        rank, model, config, metrics, transfer_outputs
    )

    eos_train = EpochOutputStore(output_transform=transfer_outputs_eos)
    eos_train.attach(train_evaluator, "output")
    eos_val = EpochOutputStore(output_transform=transfer_outputs_eos)
    eos_val.attach(val_evaluator, "output")

    history = {
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
    }

    @trainer.on(Events.EPOCH_COMPLETED)
    def _eval(engine):
        n_train_eval = int(0.1 * len(train_loader))
        # n_train_eval = len(train_loader)
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
        # console logging
        train_evaluator.add_event_handler(Events.COMPLETED, log_results, name="train")
        val_evaluator.add_event_handler(Events.COMPLETED, log_results, name="val")

        # metric logging
        val_evaluator.add_event_handler(
            Events.COMPLETED, log_metric_history, Path(config["output_dir"])
        )

        # # ray tune reporting
        if ray.tune.is_session_enabled():
            val_evaluator.add_event_handler(
                Events.COMPLETED, lambda engine: session.report(engine.state.metrics)
            )

    print("starting training loop")
    trainer.run(train_loader, max_epochs=config["epochs"])

    return val_evaluator.state.metrics["loss"]


def run_lr(local_rank: int, config):
    rank = idist.get_rank()
    print(f"hello from process {rank=}")
    config["checkpoint"] = False
    manual_seed(config["random_seed"] + local_rank)

    train_loader, val_loader = get_dataflow(config)
    model, optimizer, scheduler = _initialize(config, len(train_loader))
    scheduler = None  # explicitly disable scheduler for LRFinder
    trainer = setup_trainer(rank, model, optimizer, scheduler, config)

    lr_finder = FastaiLRFinder()
    to_save = {"model": model, "optimizer": optimizer}
    with lr_finder.attach(
        trainer,
        to_save,
        start_lr=1e-6,
        end_lr=0.1,
        num_iter=400,
        diverge_th=1e9,
    ) as finder:
        finder.run(train_loader)

    if rank == 0:
        # print("Suggested LR", lr_finder.lr_suggestion())
        ax = lr_finder.plot(display_suggestion=False)
        ax.loglog()
        ax.set_ylim(None, 5.0)
        ax.figure.savefig("lr.png")


@cli.command()
def train(config_path: Path):
    """NFF training entry point."""
    with idist.Parallel(**spawn_kwargs) as parallel:
        config = ConfigObject(config_path)
        print(config)
        parallel.run(run_train, config)


@cli.command()
def lr(config_path: Path):
    """NFF Learning rate finder entry point."""
    with idist.Parallel(**spawn_kwargs) as parallel:
        print(spawn_kwargs)
        print("loading config")
        config = ConfigObject(config_path)

        print(config)
        parallel.run(run_lr, config)


if __name__ == "__main__":
    cli()
