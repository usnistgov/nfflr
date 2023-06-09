# training utilities

## AtomsDataset

::: nfflr.data.dataset.AtomsDataset
    handler: python
    options:
      show_root_heading: true
      show_source: true


## Training scripts

`nfflr` provides a command line utility `nff` (defined in `nfflr.train.py`) to simplify common training workflows.
These use `py_config_runner` for flexible task, model, and optimization configuration.
Currently two commands are supported `nff train` for full training runs and `nff lr` for running a [learning rate finder experiment](https://pytorch.org/ignite/master/generated/ignite.handlers.lr_finder.FastaiLRFinder.html).
This training utility uses `ignite`'s auto-distributed functionality to transparently support data-parallel distributed training.

### training command

To launch a full training run:
```bash
nff train /path/to/config.py
```

The configuration file should define the task, model, and optimizer settings:

```python
# config.py
import os
from torch import nn
from pathlib import Path
from functools import partial
import ignite.distributed as idist

from nfflr.models.gnn import alignn_ff
from nfflr.data.dataset import AtomsDataset
from nfflr.data.graph import periodic_radius_graph

# set up dataset and task
experiment_dir = Path(__file__).parent.resolve()
random_seed = 42
num_workers = 6
progress = False
output_dir = experiment_dir

# set up optimizer
optimizer = "adamw"
criterion = nn.MSELoss()
epochs = 100
batch_size = 4 * idist.get_world_size()  # total batch size
weight_decay = 1e-5
learning_rate = 3e-4
warmup_steps = 2000

# data source and task specification
# source can be json file, jarvis dataset name, or folder
# task can be "energy_and_forces" or a key in the dataset
cutoff = 8.0
tfm = partial(periodic_radius_graph, r=cutoff)


data_source = Path("/wrk/bld/alignn-ff-chips/experiments/mpf-gap0.05-3/mpf-gap0.05-3-subset.jsonl")
target = "total_energy"
# target = "energy_and_forces"
energy_units = "eV"
n_train = 0.8
n_val = 0.1

dataset = AtomsDataset(
    data_source,
    target,
    energy_units=energy_units,
    n_train=n_train,
    n_val=n_val,
    transform=tfm,
    diskcache=True,
)

# set up model
model_cfg = alignn_ff.ALIGNNConfig(
    cutoff=cutoff,
    cutoff_onset=4.0,
    alignn_layers=2,
    gcn_layers=2,
    atom_features="embedding",
    compute_forces=target == "energy_and_forces",
)
model = alignn_ff.ALIGNN(model_cfg)



```
