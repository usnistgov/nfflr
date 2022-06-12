"""Prototype training code for force field models."""
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
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.handlers.stores import EpochOutputStore
from ignite.metrics import Accuracy, Loss, MeanAbsoluteError
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from alignn.config import TrainingConfig
from alignn.dataset import AtomisticConfigurationDataset
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from alignn.training_utils import (
    group_decay,
    setup_evaluator_with_grad,
    setup_optimizer,
)

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
        pct_start=pct_start,
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
        pred, tgt = batch

        axes[0].scatter(
            tgt["total_energy"].cpu().detach().numpy(),
            pred["total_energy"].cpu().detach().numpy(),
            color="k",
        )
        axes[1].scatter(
            tgt["forces"].cpu().detach().numpy(),
            pred["forces"].cpu().detach().numpy(),
            color="k",
        )

    plt.tight_layout()
    plt.savefig(Path(directory) / f"parity_plots_{phase}_{epoch:03d}.png")
    plt.clf()
    plt.close()


def train_ff(config, model, dataset):
    """Train force field model."""
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
        sampler=SubsetRandomSampler(dataset.split["train"]),
        pin_memory=True,
    )

    prepare_batch = partial(
        train_loader.dataset.prepare_batch, device=device, non_blocking=True
    )

    params = group_decay(model)
    optimizer = setup_optimizer(params, cfg)
    scheduler = setup_scheduler(cfg, optimizer, len(train_loader))

    criteria = {"total_energy": nn.MSELoss(), "forces": nn.MSELoss()}

    def ff_criterion(outputs, targets):
        """Specify combined energy and force loss."""
        energy_loss = criteria["total_energy"](
            outputs["total_energy"], targets["total_energy"]
        )

        # scale the forces before the loss
        force_scale = 10.0
        force_loss = criteria["forces"](
            force_scale * outputs["forces"], force_scale * targets["forces"]
        )

        return energy_loss + force_loss

    trainer = create_supervised_trainer(
        model,
        optimizer,
        ff_criterion,
        prepare_batch=prepare_batch,
        device=device,
    )

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

        train_evaluator.run(train_loader)
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
                train_evaluator.state.output,
                epoch,
                config.output_dir,
                phase=phase,
            )

            for key, value in m.items():
                history[phase][key].append(value)

        torch.save(history, Path(config.output_dir) / "metric_history.pkl")

    # train the model!
    trainer.run(train_loader, max_epochs=cfg.epochs)

    print(history)


if __name__ == "__main__":

    # load data from this json format into pandas...
    # then build graphs on each access instead of precomputing them...
    # also, make sure to split sections grouped on id column
    # example_data = Path("alignn/examples/sample_data")
    # df = pd.read_json(example_data / "id_prop.json")

    # jdft_trajectories = Path(
    #     "/wrk/knc6/AlIGNN-FF/jdft_max_min_307113_epa/DataDir"
    # )
    # df = pd.read_json(jdft_trajectories / "id_prop.json")
    df = pd.read_pickle("jdft_prototyping_trajectories_10k.pkl")

    model_cfg = ALIGNNAtomWiseConfig(
        name="alignn_atomwise",
        alignn_layers=2,
        gcn_layers=2,
        atom_input_features=1,
        sparse_atom_embedding=True,
        calculate_gradient=True,
    )
    # need to pass model config as dict?
    cfg = TrainingConfig(
        model=model_cfg,
        atom_features="atomic_number",
        num_workers=4,
        epochs=10,
        batch_size=256,
        output_dir="./temp",
    )
    print(cfg)

    model = ALIGNNAtomWise(model_cfg)
    model.fc.bias.data = torch.tensor([-3.0])
    model.to(device)

    lg = True
    dataset = AtomisticConfigurationDataset(
        df,
        line_graph=lg,
    )

    train_ff(cfg, model, dataset)
