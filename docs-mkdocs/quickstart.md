# Installation

Until NFFLR is registered on PyPI, it's best to install directly from github.

We recommend using a per-project [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) environment.

#### Method 1 (using setup.py):

Now, let's install the package:
```
git clone https://github.com/usnistgov/nfflr
cd nfflr
python setup.py develop
```
For using GPUs/CUDA, install dgl-cu101 or dgl-cu111 based on the CUDA version available on your system, e.g.

```
pip install dgl-cu111
```

#### Method 2 (using pypi):

Alternatively, install NFFLR directly from github using `pip`:
```
python -m pip install https://github.com/usnistgov/nfflr
```

# Examples

Load an `AtomsDataset` by name from the Jarvis figshare collection:
``` py
from nfflr.data.dataset import AtomsDataset
d = AtomsDataset("dft_2d", target="formation_energy_peratom")
```

```
dataset_name='dft_2d'
Obtaining 2D dataset 1.1k ...
Reference:https://www.nature.com/articles/s41524-020-00440-1
Other versions:https://doi.org/10.6084/m9.figshare.6815705
Loading the zipfile...
Loading completed.
```


`AtomsDataset` is a [PyTorch DataSet](https://pytorch.org/docs/stable/data.html); by default accessing a dataset item returns a tuple containing an `Atoms` instance and a target value (in this case `"formation_energy_peratom"`).
`Atoms` consists of the cell matrix, the fractional coordinates, and the atomic numbers.

``` py
d[0]
```

```
(Atoms(lattice=tensor([[ 2.8173, -0.0000,  0.0000],
         [-1.4087,  2.4399,  0.0000],
         [ 0.0000,  0.0000, 24.5518]]), positions=tensor([[0.2691, 0.9290, 0.0927],
         [0.6024, 0.5957, 0.1308],
         [0.9357, 0.2623, 0.0546]]), numbers=tensor([27,  8,  8], dtype=torch.int32)),
 tensor(-0.9247))
```


# How to contribute

We gladly accept [pull requests](https://makeapullrequest.com).
