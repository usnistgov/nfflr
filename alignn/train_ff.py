"""Prototype training code for force field models."""
from typing import Any, Callable

import pandas as pd
import torch
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Engine,
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Accuracy, Loss, MeanAbsoluteError
from torch import nn
from torch.utils.data import DataLoader

from alignn.config import TrainingConfig
from alignn.dataset import AtomisticConfigurationDataset
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from alignn.train import group_decay, setup_optimizer

# def setup_training(config, model):
#     params = group_decay(model)
#     optimizer = setup_optimizer(params, config)


def setup_evaluator(
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

        return output_transform(x, y, y_pred)

    evaluator = Engine(evaluation_step)
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


if __name__ == "__main__":
    from pathlib import Path

    example_data = Path("alignn/examples/sample_data")
    df = pd.read_json(example_data / "id_prop.json")

    model_cfg = ALIGNNAtomWiseConfig(
        name="alignn_atomwise",
        alignn_layers=2,
        gcn_layers=2,
        atom_input_features=1,
        calculate_gradient=True,
    )
    cfg = TrainingConfig(model=model_cfg, num_workers=0, epochs=10)
    print(cfg)

    model = ALIGNNAtomWise(model_cfg)

    lg = True
    dataset = AtomisticConfigurationDataset(
        df,
        line_graph=lg,  # atom_features="cgcnn"
    )
    dl = DataLoader(
        dataset, collate_fn=dataset.collate, batch_size=16, drop_last=True
    )
    prepare_batch = dl.dataset.prepare_batch

    params = group_decay(model)
    optimizer = setup_optimizer(params, cfg)

    steps_per_epoch = len(dl)
    pct_start = cfg.warmup_steps / (cfg.epochs * steps_per_epoch)
    pct_start = min(pct_start, 0.3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.learning_rate,
        epochs=cfg.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
    )

    criteria = {"total_energy": nn.MSELoss(), "forces": nn.MSELoss()}

    def ff_criterion(outputs, targets):
        """Specify combined energy and force loss."""
        energy_loss = criteria["total_energy"](
            outputs["total_energy"], targets["total_energy"]
        )

        force_loss = criteria["forces"](outputs["forces"], targets["forces"])

        return energy_loss + force_loss

    metrics = {
        "loss": Loss(ff_criterion),
        # "mae": MeanAbsoluteError()
    }

    trainer = create_supervised_trainer(
        model,
        optimizer,
        ff_criterion,
        prepare_batch=prepare_batch,
        # device=device,
    )

    # create_supervised_evaluator(
    train_evaluator = setup_evaluator(
        model,
        metrics=metrics,
        prepare_batch=prepare_batch,
        # device=device,
    )

    # train_evaluator.add_event_handler(
    #     Events.ITERATION_STARTED, lambda _: torch.set_grad_enabled(True)
    # )

    # ignite event handlers:
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    # apply learning rate scheduler
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )

    pbar = ProgressBar()
    pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        """Log training results."""
        train_evaluator.run(dl)
        metrics = train_evaluator.state.metrics
        epoch, loss = trainer.state.epoch, metrics["loss"]
        print(f"Training Results - Epoch: {epoch}  Avg loss: {loss:.2f}")

    # train the model!
    trainer.run(dl, max_epochs=cfg.epochs)
