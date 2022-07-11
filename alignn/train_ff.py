"""Prototype training code for force field models."""
import typer
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from ignite.contrib.engines.common import setup_common_training_handlers
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    FastaiLRFinder,
    TerminateOnNan,
)
from ignite.handlers.stores import EpochOutputStore
from ignite.metrics import Accuracy, Loss, MeanAbsoluteError
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from alignn.config import TrainingConfig
from alignn.dataset import AtomisticConfigurationDataset
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from alignn.models.bond_order import BondOrderConfig, NeuralBondOrder
from alignn.training_utils import (
    group_decay,
    setup_evaluator_with_grad,
    setup_optimizer,
)

cli = typer.Typer()

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def setup_scheduler(config, optimizer, steps_per_epoch):
    """Configure OneCycle scheduler."""
    pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
    pct_start = min(pct_start, 0.3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        # pct_start=pct_start,
    )

    return scheduler


def select_target(name: str):
    """Build ignite metric transforms for multi-output models.

    `output` should be Tuple[Dict[str,torch.Tensor], Dict[str,torch.Tensor]]
    """

    def output_transform(output):
        """Select output tensor for metric computation."""
        pred, target = output
        return pred[name], target[name]

    return output_transform


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


def get_dataflow(config):
    """Set up dataloaders."""
    # load data from this json format into pandas...
    # then build graphs on each access instead of precomputing them...
    # also, make sure to split sections grouped on id column
    # example_data = Path("alignn/examples/sample_data")
    # df = pd.read_json(example_data / "id_prop.json")

    # _epa suffix has energies per atom...
    # dataset = "jdft_max_min_307113_epa"
    # dataset = "jdft_max_min_307113"
    dataset = "jdft_max_min_307113_id_prop.json"

    datadir = Path("data")
    df = pd.read_json(datadir / dataset)
    df = df.iloc[:10000]

    dataset = AtomisticConfigurationDataset(
        df,
        line_graph=True,
        energy_units="eV/atom",
    )

    # df = pd.read_json("jdft_prototyping_trajectories_10k.jsonl", lines=True)
    print(df.shape)

    train_loader = DataLoader(
        dataset,
        collate_fn=dataset.collate,
        batch_size=config.batch_size,
        sampler=SubsetRandomSampler(dataset.split["train"]),
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset,
        collate_fn=dataset.collate,
        batch_size=config.batch_size,
        sampler=SubsetRandomSampler(dataset.split["val"]),
        pin_memory=True,
    )

    return train_loader, val_loader


@cli.command()
def train():
    """Train force field model."""

    model_cfg = ALIGNNAtomWiseConfig(
        name="alignn_atomwise",
        alignn_layers=2,
        gcn_layers=2,
        atom_input_features=1,
        sparse_atom_embedding=True,
        calculate_gradient=True,
    )
    config = TrainingConfig(
        model=model_cfg,
        atom_features="atomic_number",
        num_workers=8,
        epochs=100,
        batch_size=128,
        learning_rate=0.001,
        output_dir="./models/ff-300k",
    )

    model = ALIGNNAtomWise(config.model)
    model.to(device)

    train_loader, val_loader = get_dataflow(config)

    prepare_batch = partial(
        train_loader.dataset.prepare_batch, device=device, non_blocking=True
    )

    params = group_decay(model)
    optimizer = setup_optimizer(params, config)
    scheduler = setup_scheduler(config, optimizer, len(train_loader))

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

    # pbar = ProgressBar()
    # pbar.attach(trainer, output_transform=lambda x: {"loss": x})

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
        DiskSaver(config.output_dir, create_dir=True, require_empty=False),
        n_saved=1,
        global_step_transform=lambda *_: trainer.state.epoch,
    )
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

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        """Log training results."""
        epoch = engine.state.epoch

        n_train_eval = int(0.1 * len(train_loader))
        train_evaluator.run(
            train_loader, epoch_length=n_train_eval, max_epochs=1
        )
        val_evaluator.run(val_loader)

        evaluators = {"train": train_evaluator, "validation": val_evaluator}

        for phase, evaluator in evaluators.items():

            m = evaluator.state.metrics
            loss = m["loss"]

            print(f"{phase} results - Epoch: {epoch}  Avg loss: {loss:.2f}")
            print(
                f"energy: {m['mae_energy']:.2f}  force: {m['mae_forces']:.4f}"
            )

            parity_plots(
                evaluator.state.output,
                epoch,
                config.output_dir,
                phase=phase,
            )

            for key, value in m.items():
                history[phase][key].append(value)

        torch.save(history, Path(config.output_dir) / "metric_history.pkl")

    # train the model!
    trainer.run(train_loader, max_epochs=config.epochs)

    print(history)


@cli.command()
def lr():
    """run learning rate finder."""
    # torch.autograd.set_detect_anomaly(True)

    torch.set_default_dtype(torch.float64)  # batch size=64

    model_cfg = ALIGNNAtomWiseConfig(
        name="alignn_atomwise",
        alignn_layers=2,
        gcn_layers=2,
        atom_input_features=1,
        sparse_atom_embedding=True,
        calculate_gradient=True,
    )
    config = TrainingConfig(
        model=model_cfg,
        atom_features="atomic_number",
        num_workers=8,
        epochs=100,
        batch_size=64,
        learning_rate=0.001,
        output_dir="./models/ff-300k",
    )

    model = ALIGNNAtomWise(config.model)
    model.to(device)

    train_loader, val_loader = get_dataflow(config)

    prepare_batch = partial(
        train_loader.dataset.prepare_batch, device=device, non_blocking=True
    )

    params = group_decay(model)

    optimizer = setup_optimizer(params, config)
    scheduler = setup_scheduler(config, optimizer, len(train_loader))

    criteria = {
        "total_energy": nn.MSELoss(),
        "forces": nn.HuberLoss(delta=0.1),
    }

    def ff_criterion(outputs, targets):
        """Specify combined energy and force loss."""
        energy_loss = criteria["total_energy"](
            outputs["total_energy"], targets["total_energy"]
        )

        # print("energy_loss", torch.isnan(energy_loss).any())

        # # scale the forces before the loss
        force_scale = 1.0
        force_loss = criteria["forces"](outputs["forces"], targets["forces"])
        # print("force_loss", torch.isnan(force_loss).any())

        return energy_loss + force_scale * force_loss

    trainer = create_supervised_trainer(
        model,
        optimizer,
        ff_criterion,
        prepare_batch=prepare_batch,
        device=device,
    )

    pbar = ProgressBar()
    pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    lr_finder = FastaiLRFinder()
    to_save = {"model": model, "optimizer": optimizer}
    with lr_finder.attach(
        trainer,
        to_save,
        start_lr=1e-6,
        end_lr=1.0,
        num_iter=400,
    ) as finder:
        finder.run(train_loader)

    print("Suggested LR", lr_finder.lr_suggestion())
    ax = lr_finder.plot()
    ax.loglog()
    ax.figure.savefig("lr.png")


if __name__ == "__main__":
    cli()
