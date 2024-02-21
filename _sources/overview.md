# Overview

## input representation
For efficient training, models should also be able to operate on preprocessed structures,
e.g. a graph neural network could implement `forward(x: dgl.DGLGraph)` to allow asynchronous
construction of graph batches in a `DataLoader`.

- [dgl.DGLGraph](#dgl.DGLGraph)
- PyG input format
- `nfflr.Atoms` with ghost atom padding
- some other custom input representation


## modeling interface

Models should have a consistent interface, regardless of the backend!

```
NFFLrModel:
    forward(x: Atoms) -> dict[str, torch.Tensor]
```

Depending on the model and the prediction task, both the input representation and output format may vary.



### output representation
Depending on the task, a model may return predictions in different formats.

- Single-target tasks like scalar or vector regression or classification naturally return a single tensor `forward(x: Atoms) -> torch.Tensor`.
- A force-field should return a `dict[str, torch.Tensor]`: `{"total_energy": torch.Tensor, "forces": torch.Tensor, "stress": torch.Tensor}`, where `total_energy` is a scalar, `forces` is an `(n_atoms, n_spatial_dimensions)` tensor, and `stress` is a `(n_spatial_dimensions, n_spatial_dimensions)` tensor
- A custom multi-target task might also return a `dict` of


## [ignite](#ignite)-based trainer

`nfflr` includes a command line utility `nff` (defined in `nfflr.train.trainer.py`) to simplify common training workflows.
These use `py_config_runner` for flexible task, model, and optimization configuration.
Currently two commands are supported: `nff train` for full training runs and `nff lr` for running a [learning rate finder experiment](https://pytorch.org/ignite/master/generated/ignite.handlers.lr_finder.FastaiLRFinder.html).
This training utility uses [ignite](#ignite)'s auto-distributed functionality to transparently support data-parallel distributed training.

### training command

To launch a full training run:
```bash
nff train /path/to/config.py
```

The configuration file should define the task, model, and optimizer settings:

```python
# config.py
from torch import nn
from pathlib import Path
from functools import partial

import nfflr

trainer = nfflr.train.TrainingConfig(
    experiment_dir=Path(__file__).parent.resolve(),
    random_seed=42,
    dataloader_workers=0,
    optimizer="adamw",
    epochs=20,
    per_device_batch_size=16,
    weight_decay=1e-3,
    learning_rate=3e-3,
)

# data source and task specification
criterion = nfflr.nn.MultitaskLoss(["energy", "forces"])

rcut = 4.25
transform = nfflr.nn.PeriodicRadiusGraph(cutoff=rcut)
cutoff = nfflr.nn.Cosine(rcut)
model_cfg = nfflr.models.ALIGNNFFConfig(
    transform=transform,
    cutoff=cutoff,
    atom_features="embedding",
    alignn_layers=2,
    gcn_layers=2,
    compute_forces=True,
    energy_units="eV",
)
model = nfflr.models.ALIGNNFF(model_cfg)

# source can be json file, jarvis dataset name, or folder
# task can be "energy_and_forces" or a key in the dataset
dataset = nfflr.data.mlearn_dataset(transform=tfm)
```
