"""Prototype training code for force field models."""
import pandas as pd
import torch
from ignite.engine import (
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
    cfg = TrainingConfig(model=model_cfg, num_workers=0)
    print(cfg)

    model = ALIGNNAtomWise(model_cfg)

    lg = True
    dataset = AtomisticConfigurationDataset(
        df,
        line_graph=lg,  # atom_features="cgcnn"
    )
    dl = DataLoader(dataset, collate_fn=dataset.collate, batch_size=4)
    prepare_batch = dl.dataset.prepare_batch

    params = group_decay(model)
    optimizer = setup_optimizer(params, cfg)

    steps_per_epoch = len(dl)
    pct_start = cfg.warmup_steps / (cfg.epochs * steps_per_epoch)
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

    metrics = {"loss": Loss(ff_criterion), "mae": MeanAbsoluteError()}

    trainer = create_supervised_trainer(
        model,
        optimizer,
        ff_criterion,
        prepare_batch=prepare_batch,
        # device=device,
    )

    train_evaluator = create_supervised_evaluator(
        model,
        metrics=metrics,
        prepare_batch=prepare_batch,
        # device=device,
    )

    # ignite event handlers:
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    # apply learning rate scheduler
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )

    # train the model!
    trainer.run(dl, max_epochs=cfg.epochs)
