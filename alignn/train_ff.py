"""Prototype training code for force field models."""
import pandas as pd
import torch
from torch.utils.data import DataLoader

from alignn.config import TrainingConfig
from alignn.dataset import AtomisticConfigurationDataset
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig

if __name__ == "__main__":
    from pathlib import Path

    example_data = Path("alignn/examples/sample_data")
    df = pd.read_json(example_data / "id_prop.json")

    model_cfg = ALIGNNAtomWiseConfig(
        name="alignn_atomwise", alignn_layers=2, gcn_layers=2
    )
    cfg = TrainingConfig(model=model_cfg)

    model = ALIGNNAtomWise(model_cfg)

    dataset = AtomisticConfigurationDataset(df, line_graph=False)
    dl = DataLoader(dataset, collate_fn=dataset.collate, batch_size=4)
