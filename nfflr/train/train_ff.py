"""Prototype training code for force field models."""
__all__ = ()
import os
from datetime import timedelta
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, Optional, Union

from ray.air import session

import ignite.distributed as idist
import matplotlib.pyplot as plt
import pandas as pd
import torch
import typer
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_trainer,
)
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    FastaiLRFinder,
    TerminateOnNan,
)
from ignite.handlers.stores import EpochOutputStore
from ignite.metrics import Loss, MeanAbsoluteError
from ignite.utils import manual_seed
from jarvis.db.figshare import data as jdata
from torch import nn
from torch.utils.data import SubsetRandomSampler

from alignn.dataset import AtomisticConfigurationDataset
from alignn.models.alignn_ff import ALIGNNForceField, ALIGNNForceFieldConfig
from alignn.models.bond_order import NeuralBondOrder
from alignn.training_utils import (
    group_decay,
    select_target,
    setup_evaluator_with_grad,
    setup_optimizer,
    setup_scheduler,
)

# torch.autograd.set_detect_anomaly(True)

cli = typer.Typer()

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
num_gpus = len(gpus.split(","))


# https://pytorch.org/docs/stable/distributed.html#which-backend-to-use
distributed_backend = None
if torch.distributed.is_nccl_available():
    # prefer nccl if available for GPU parallelism
    distributed_backend = "nccl"
elif torch.distributed.is_gloo_available():
    distributed_backend = "gloo"

# distributed_backend = "gloo"


@dataclass
class DatasetConfig:
    name: Union[Literal["alignn_ff_db"], Path]
    atom_features: Literal["basic", "atomic_number", "cfid", "cgcnn"] = "atomic_number"
    random_seed: Optional[int] = 123
    n_val: Optional[Union[int, float]] = 0.1
    n_train: Optional[Union[int, float]] = 0.8
    num_workers: int = 4
    # jarvis_cache: Path = Path("~/.jarvis").expanduser()
    jarvis_cache: Path = Path("/wrk/bld/shared/jarvis").expanduser()


@dataclass
class OptimizerConfig:
    optimizer: Literal["sgd", "adamw"] = "adamw"
    batch_size: int = 256
    learning_rate: float = 1e-2
    weight_decay: float = 1e-5
    epochs: int = 30
    warmup_steps: int = 2000
    progress: bool = True
    output_dir: Path = Path(".")


@dataclass
class FFConfig:
    dataset: DatasetConfig
    optimizer: OptimizerConfig
    model: Any


def ff_config(config: dict):
    return FFConfig(
        dataset=DatasetConfig(**config["dataset"]),
        optimizer=OptimizerConfig(**config["optimizer"]),
        model=ALIGNNForceFieldConfig(**config["model"]),
    )


def get_dataflow(config):
    """Set up force field dataloaders."""
    # _epa suffix has energies per atom...
    # dataset = "jdft_max_min_307113_epa"
    # dataset = "jdft_max_min_307113"
    # dataset = "jdft_max_min_307113_id_prop.json"
    # datadir = Path("data")

    if isinstance(config.dataset.name, Path):
        # e.g., "jdft_max_min_307113_id_prop.json"
        lines = "jsonl" in config.dataset.name
        df = pd.read_json(config.dataset.name, lines=lines)

    elif config.dataset.name == "alignn_ff_db":
        df = pd.DataFrame(
            jdata(config.dataset.name, store_dir=config.dataset.jarvis_cache)
        )

    # in a distributed setting, ensure only the rank 0 process
    # creates any lmdb store on disk
    if idist.get_local_rank() > 0:
        idist.barrier()

    dataset = AtomisticConfigurationDataset(
        df,
        line_graph=False,
        cutoff_radius=config.model.cutoff,
        neighbor_strategy="cutoff",
        energy_units="eV/atom",
        n_train=config.dataset.n_train,
        n_val=config.dataset.n_val,
    )

    # rank 0 triggers lmdb connect/write
    _ = dataset[0]

    # and now non-root processes can continue
    if idist.get_local_rank() == 0:
        idist.barrier()

    # configure training and validation datasets...
    train_loader = idist.auto_dataloader(
        dataset,
        collate_fn=dataset.collate,
        batch_size=config.optimizer.batch_size,
        sampler=SubsetRandomSampler(dataset.split["train"]),
        drop_last=True,
        num_workers=config.dataset.num_workers,
        pin_memory="cuda" in idist.device().type,
    )
    val_loader = idist.auto_dataloader(
        dataset,
        collate_fn=dataset.collate,
        batch_size=config.optimizer.batch_size,
        sampler=SubsetRandomSampler(dataset.split["val"]),
        num_workers=config.dataset.num_workers,
        pin_memory="cuda" in idist.device().type,
    )

    return train_loader, val_loader


@cli.command()
def train():
    """Train force field model."""

    model_cfg = ALIGNNForceFieldConfig(
        name="alignn_forcefield",
        cutoff=8.0,
        cutoff_onset=7.5,
        alignn_layers=2,
        gcn_layers=2,
        atom_input_features=1,
        sparse_atom_embedding=True,
        calculate_gradient=True,
    )

    # model_cfg = BondOrderConfig(
    #     name="bondorder",
    # cutoff=8.0,
    # cutoff_onset=7.5,
    #     alignn_layers=2,
    #     gcn_layers=2,
    #     calculate_gradient=True,
    # )

    data_cfg = DatasetConfig(
        name="alignn_ff_db",
        n_train=50000,
        n_val=1000,
        num_workers=6,
    )

    opt_cfg = OptimizerConfig(
        batch_size=64 * num_gpus,  # 128 * 4
        weight_decay=1e-1,
        learning_rate=1e-3,
        progress=True,
        output_dir="./ff-300k-dist",
    )

    config = FFConfig(model=model_cfg, optimizer=opt_cfg, dataset=data_cfg)

    # this can't be None if training distributed...
    spawn_kwargs = {
        "nproc_per_node": num_gpus,
        "timeout": timedelta(seconds=60),
    }
    # spawn_kwargs = {}

    print("launching...")
    with idist.Parallel(backend=distributed_backend, **spawn_kwargs) as parallel:
        parallel.run(run_train, config)


def run_train(local_rank, config):
    if isinstance(config, dict):
        config = ff_config(config)

    # torch.set_default_dtype(torch.float64)  # batch size=64

    rank = idist.get_rank()
    manual_seed(config.dataset.random_seed + rank)
    device = idist.device()

    print(f"running training {config.model.name} on {device}")

    if config.model.name == "alignn_forcefield":
        model = idist.auto_model(ALIGNNForceField(config.model))
    elif config.model.name == "bondorder":
        model = idist.auto_model(NeuralBondOrder(config.model))

    train_loader, val_loader = get_dataflow(config)
    prepare_batch = partial(
        train_loader.dataset.prepare_batch, device=device, non_blocking=True
    )

    params = group_decay(model)
    optimizer = setup_optimizer(params, config.optimizer)
    optimizer = idist.auto_optim(optimizer)
    scheduler = setup_scheduler(config.optimizer, optimizer, len(train_loader))

    criteria = {
        "total_energy": nn.MSELoss(),
        "forces": nn.HuberLoss(delta=0.1),
    }

    def ff_criterion(outputs, targets):
        """Specify combined energy and force loss."""
        energy_loss = criteria["total_energy"](
            outputs["total_energy"], targets["total_energy"]
        )

        # # scale the forces before the loss
        force_scale = 1.0
        force_loss = criteria["forces"](outputs["forces"], targets["forces"])

        return energy_loss + force_scale * force_loss

    trainer = create_supervised_trainer(
        model,
        optimizer,
        ff_criterion,
        prepare_batch=prepare_batch,
        device=device,
    )

    if config.optimizer.progress and rank == 0:
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
        DiskSaver(config.optimizer.output_dir, create_dir=True, require_empty=False),
        n_saved=1,
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    if rank == 0:
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)

    # evaluation
    metrics = {
        "loss": Loss(ff_criterion),
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

    # train_evaluator.add_event_handler(
    #     Events.ITERATION_COMPLETED, lambda engine: print("evaluation step")
    # )

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

        session.report(checkpoint_results)

        for phase, evaluator in evaluators.items():
            m = evaluator.state.metrics
            loss = m["loss"]

            print(f"{phase} results - Epoch: {epoch}  Avg loss: {loss:.2f}")
            print(f"energy: {m['mae_energy']:.2f}  force: {m['mae_forces']:.4f}")

            parity_plots(
                evaluator.state.output,
                epoch,
                config.optimizer.output_dir,
                phase=phase,
            )

            for key, value in m.items():
                history[phase][key].append(value)

        torch.save(history, Path(config.optimizer.output_dir) / "metric_history.pkl")

    # train the model!
    print("go")
    trainer.run(train_loader, max_epochs=config.optimizer.epochs)

    return val_evaluator.state.metrics["loss"]


@cli.command()
def lr():
    data_cfg = DatasetConfig(
        name="alignn_ff_db",
        n_train=1000,
        n_val=1000,
        num_workers=6,
    )

    opt_cfg = OptimizerConfig(
        batch_size=100,
        weight_decay=1e-1,
        learning_rate=1e-2,
        progress=True,
        output_dir="./models/ff-300k",
    )

    model_cfg = ALIGNNForceFieldConfig(
        name="alignn_forcefield",
        cutoff=8.0,
        cutoff_onset=7.5,
        alignn_layers=2,
        gcn_layers=2,
        atom_input_features=1,
        sparse_atom_embedding=True,
        calculate_gradient=True,
    )
    # model_cfg = BondOrderConfig(
    #     name="bondorder",
    #     cutoff=8.0,
    #     cutoff_onset=7.5,
    #     alignn_layers=2,
    #     gcn_layers=2,
    #     calculate_gradient=True,
    # )

    config = FFConfig(dataset=data_cfg, optimizer=opt_cfg, model=model_cfg)

    # spawn_kwargs = {"nproc_per_node": 2}
    spawn_kwargs = {}

    print("launching...")
    with idist.Parallel(backend=distributed_backend, **spawn_kwargs) as parallel:
        parallel.run(run_lr, config)


def run_lr(local_rank, config):
    """run learning rate finder."""
    # torch.set_default_dtype(torch.float64)  # batch size=64

    rank = idist.get_rank()
    manual_seed(config.dataset.random_seed + rank)
    device = idist.device()
    print(f"running lr finder on {device}")

    if config.model.name == "alignn_forcefield":
        model = ALIGNNForceField(config.model)
    elif config.model.name == "bondorder":
        model = NeuralBondOrder(config.model)

    idist.auto_model(model)

    train_loader, val_loader = get_dataflow(config)
    prepare_batch = partial(
        train_loader.dataset.prepare_batch, device=device, non_blocking=True
    )

    optimizer = setup_optimizer(group_decay(model), config.optimizer)
    optimizer = idist.auto_optim(optimizer)

    # scheduler = setup_scheduler(config, optimizer, len(train_loader))

    criteria = {
        "total_energy": nn.MSELoss(),
        "forces": nn.HuberLoss(delta=0.1),
    }

    def ff_criterion(outputs, targets):
        """Specify combined energy and force loss."""
        energy_loss = criteria["total_energy"](
            outputs["total_energy"], targets["total_energy"]
        )

        # # scale the forces before the loss
        force_scale = 1.0
        force_loss = criteria["forces"](outputs["forces"], targets["forces"])

        return energy_loss + force_scale * force_loss

    # scaler = torch.cuda.amp.GradScaler()
    # update_fn = supervised_training_step_amp(model, optimizer, ff_criterion, 'cuda', scaler=scaler, prepare_batch=prepare_batch)
    # trainer = Engine(update_fn)

    trainer = create_supervised_trainer(
        model,
        optimizer,
        ff_criterion,
        prepare_batch=prepare_batch,
        device=device,
    )

    if rank == 0:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    print("go")
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
        print("Suggested LR", lr_finder.lr_suggestion())
        ax = lr_finder.plot()
        ax.loglog()
        ax.figure.savefig("lr.png")


def parity_plots(output, epoch, directory, phase="train"):
    """Plot predictions for energy and forces."""
    fig, axes = plt.subplots(ncols=2, figsize=(16, 8))
    for batch in output:
        tgt, pred = batch

        axes[0].scatter(
            tgt["total_energy"].cpu().detach().numpy(),
            pred["total_energy"].cpu().detach().numpy(),
            color="k",
        )
        axes[0].set(xlabel="DFT energy", ylabel="predicted energy")
        axes[1].scatter(
            tgt["forces"].cpu().detach().numpy(),
            pred["forces"].cpu().detach().numpy(),
            color="k",
        )
        axes[1].set(xlabel="DFT force", ylabel="predicted force")

    plt.tight_layout()
    plt.savefig(Path(directory) / f"parity_plots_{phase}_{epoch:03d}.png")
    plt.clf()
    plt.close()


if __name__ == "__main__":
    cli()
