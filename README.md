# NFFLr - Neural Force Field Learning toolkit
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/usnistgov/nfflr)
![GitHub commit activity](https://img.shields.io/github/commit-activity/y/usnistgov/nfflr)

# Table of Contents
* [Introduction](#intro)
* [Installation](#install)
* [Examples](#example)
* [How to contribute](#contrib)
* [Correspondence](#corres)
* [Funding support](#fund)

<a name="intro"></a>
# NFFLr (Introduction)
The Neural Force Field Learning library ([docs](https://pages.nist.gov/nfflr/index.html)) is intended to be a flexible toolkit for developing and deploying atomistic machine learning systems, with a particular focus on crystalline material property and energy models.

The initial codebase is a fork of [ALIGNN](https://github.com/usnistgov/alignn), with modified configuration and modeling interfaces for performance.

<a name="install"></a>
Installation
-------------------------
Until NFFLr is registered on PyPI, it's best to install directly from github.

We recommend using a per-project [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) environment.

#### Method 1 (using setup.py):

Now, let's install the package:
```
git clone https://github.com/usnistgov/nfflr
cd nfflr
python -m pip install -e .
```
For using GPUs/CUDA, install dgl-cu101 or dgl-cu111 based on the CUDA version available on your system, e.g.

```
pip install dgl-cu111
```

#### Method 2 (using pypi):

Alternatively, install NFFLr directly from github using `pip`:
```
python -m pip install https://github.com/usnistgov/nfflr
```

<a name="example"></a>
Examples
---------
[Under construction here.](https://pages.nist.gov/nfflr/examples.html)


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
