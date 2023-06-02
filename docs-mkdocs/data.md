# Data

The primary ways of interacting with data are `Atoms` and `AtomsDataset`,
which is a [PyTorch DataSet](https://pytorch.org/docs/stable/data.html)

```python
from nfflr.data.dataset import AtomsDataset
d = AtomsDataset("dft_2d", target="formation_energy_peratom")
```

```shell
dataset_name='dft_2d'
Obtaining 2D dataset 1.1k ...
Reference:https://www.nature.com/articles/s41524-020-00440-1
Other versions:https://doi.org/10.6084/m9.figshare.6815705
Loading the zipfile...
Loading completed.
```



::: nfflr.data.atoms.Atoms
    handler: python
    options:
      show_root_heading: true
      show_source: true
