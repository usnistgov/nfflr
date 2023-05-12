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

Coming soon.

# How to contribute

We gladly accept [pull requests](https://makeapullrequest.com).
