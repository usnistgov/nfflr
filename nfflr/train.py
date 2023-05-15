import os
from functools import partial
from datetime import timedelta
from pathlib import Path

from typing import Any, Dict, Tuple

import torch
from torch.utils.data import SubsetRandomSampler

import typer
import ignite.distributed as idist
from ignite.utils import manual_seed
from ignite.metrics import Loss, MeanAbsoluteError
from ignite.handlers.stores import EpochOutputStore
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import Checkpoint, DiskSaver, FastaiLRFinder, TerminateOnNan
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer


from py_config_runner import ConfigObject

from nfflr.training_utils import (
    group_decay,
    select_target,
    setup_evaluator_with_grad,
    setup_optimizer,
    setup_scheduler,
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
    model = config["model"]
    params = group_decay(model)
    optimizer = setup_optimizer(params, config)
    optimizer = idist.auto_optim(optimizer)
    scheduler = setup_scheduler(config, optimizer, steps_per_epoch)

    return model, optimizer, scheduler


def prepare_batch_dict(
    batch: Tuple[Any, Dict[str, torch.Tensor]],
    device=None,
    non_blocking=False,
) -> Tuple[Any, Dict[str, torch.Tensor]]:
    """Send batched dgl crystal graph to device."""
    atoms, targets = batch
    targets = {k: v.to(device, non_blocking=non_blocking) for k, v in targets.items()}

    batch = (atoms.to(device, non_blocking=non_blocking), targets)

    return batch


def run_train(local_rank: int, config):

    rank = idist.get_rank()
    manual_seed(config["random_seed"] + local_rank)
    device = idist.device()

    train_loader, val_loader = get_dataflow(config)
    prepare_batch = partial(prepare_batch_dict, device=device, non_blocking=True)

    model, optimizer, scheduler = _initialize(config, len(train_loader))

    trainer = create_supervised_trainer(
        model,
        optimizer,
        config["criterion"],
        prepare_batch=prepare_batch,
        device=device,
    )

    if config["progress"] and rank == 0:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    checkpoint_handler = Checkpoint(
        dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer=trainer,
        ),
        DiskSaver(config["output_dir"], create_dir=True, require_empty=False),
        n_saved=1,
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    if rank == 0:
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)

    # evaluation
    metrics = {
        "loss": Loss(config["criterion"]),
        "mae_energy": MeanAbsoluteError(select_target("total_energy")),
        "mae_forces": MeanAbsoluteError(select_target("forces")),
    }

    # def transfer_outputs(outputs):
    def transfer_outputs(x, y, y_pred):
        return tuple(
            {key: value.detach().cpu() for key, value in xs.items()}
            for xs in (y, y_pred)
        )

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

    if config["progress"] and rank == 0:
        vpbar = ProgressBar()
        vpbar.attach(train_evaluator)
        vpbar.attach(val_evaluator)

    # save outputs
    # list of outputs for each batch
    # in this case List[Tuple[Dict[str,torch.Tensor], Dict[str,torch.Tensor]]]
    def transfer_outputs_eos(outputs):
        y, y_pred = outputs
        return tuple(
            {key: value.detach().cpu() for key, value in xs.items()}
            for xs in (y, y_pred)
        )

    eos = EpochOutputStore(output_transform=transfer_outputs_eos)
    eos.attach(train_evaluator, "output")
    eos.attach(val_evaluator, "output")

    history = {
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
    }

    # TODO: refactor for readability
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        """Log training results."""
        epoch = engine.state.epoch

        n_train_eval = int(0.1 * len(train_loader))

        train_evaluator.run(train_loader, epoch_length=n_train_eval, max_epochs=1)
        val_evaluator.run(val_loader)

        if rank == 0:
            evaluators = {
                "train": train_evaluator,
                "validation": val_evaluator,
            }
        else:
            # only root process logs results
            return

        checkpoint_results = {}
        for phase, evaluator in evaluators.items():
            m = evaluator.state.metrics
            results = {f"{phase}_{key}": v for key, v in m.items()}
            checkpoint_results.update(results)

        # session.report(checkpoint_results)

        for phase, evaluator in evaluators.items():
            m = evaluator.state.metrics
            loss = m["loss"]

            print(f"{phase} results - Epoch: {epoch}  Avg loss: {loss:.2f}")
            print(f"energy: {m['mae_energy']:.2f}  force: {m['mae_forces']:.4f}")

            # parity_plots(
            #     evaluator.state.output,
            #     epoch,
            #     config["output_dir"],
            #     phase=phase,
            # )

            for key, value in m.items():
                history[phase][key].append(value)

        torch.save(history, Path(config["output_dir"]) / "metric_history.pkl")

    # train the model!
    print("go")
    trainer.run(train_loader, max_epochs=config["epochs"])

    return val_evaluator.state.metrics["loss"]


@cli.command()
def train(config_path: Path):

    with idist.Parallel(**spawn_kwargs) as parallel:
        config = ConfigObject(config_path)
        print(config)
        parallel.run(run_train, config)

    return


if __name__ == "__main__":
    cli()
