# NFFLr - Neural Force Field Learning toolkit
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/usnistgov/nfflr)
![GitHub commit activity](https://img.shields.io/github/commit-activity/y/usnistgov/nfflr)
[![General badge](https://img.shields.io/badge/docs-main-blue.svg)](https://pages.nist.gov/nfflr)


# Table of Contents
* [Introduction](#intro)
* [Installation](#install)
* [Examples](#example)
* [How to contribute](#contrib)
* [Correspondence](#corres)
* [Funding support](#fund)

<a name="intro"></a>
# NFFLr (Introduction)
The Neural Force Field Learning library ([docs](https://pages.nist.gov/nfflr)) is intended to be a flexible toolkit for developing and deploying atomistic machine learning systems, with a particular focus on crystalline material property and energy models.

The initial codebase is a fork of [ALIGNN](https://github.com/usnistgov/alignn), with modified configuration and modeling interfaces for performance.

<a name="install"></a>
Installation
------------

We recommend using a per-project [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) environment.

To ensure proper CUDA support, make sure to install the GPU versions of [PyTorch](https://pytorch.org/get-started/locally/) and [DGL](https://www.dgl.ai/pages/start.html). For example, to set up a conda environment on linux with with python 3.10 and CUDA 12.1:

```
conda create --name myproject python=3.10
conda activate myproject
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c dglteam/label/cu121 dgl
python -m pip install nfflr
```

<a name="example"></a>
Examples
--------
[Under construction here.](https://pages.nist.gov/nfflr/tutorials)


<a name="contrib"></a>
How to contribute
-----------------

We gladly accept [pull requests](https://makeapullrequest.com).

For detailed instructions, please see [Contributing.md](Contributing.md)

<a name="corres"></a>
Correspondence
--------------------

Please report bugs as Github issues (https://github.com/usnistgov/nfflr/issues) or email to brian.decost@nist.gov.

<a name="fund"></a>
Funding support
--------------------

NIST-MGI (https://www.nist.gov/mgi).

Code of conduct
--------------------

Please see [Code of conduct](https://github.com/usnistgov/jarvis/blob/master/CODE_OF_CONDUCT.md)
