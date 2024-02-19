# NFFLr documentation

```{role} python(code)
:language: python
```

Neural Force Field Learning toolkit (NFFLr) is an experimental library intended to support rapid iteration in research on machine learning for atomistic systems.
The main goal is to provide an easily extensible uniform interface for developing, training, and deploying atomistic machine learning models.

```{margin} Project history
The initial codebase is a fork of [ALIGNN](https://github.com/usnistgov/alignn), with modified configuration and modeling interfaces to enable usability and performance improvements.
```

- discuss development on [GitHub](https://github.com/usnistgov/nfflr)
- read the [documentation](https://pages.nist.gov/nfflr)
- install with `pip install nfflr`


The primary components of NFFLr are:

1. a [general atomistic representation](#nfflr.Atoms) that is easy to work with, supports efficient batching, and can be easily converted to input formats of various machine learning models.

2. a [consistent modeling interface](reference/models), regardless of backend or material representation.

3. a [trainer](reference/trainer) (based on [ignite](#ignite))that can be used regardless of the model and/or backend.




```{toctree}
:maxdepth: 3
:glob:
:hidden:
overview
tutorials/index
howto/index
reference/index
```

## Correspondence

Please report bugs by filing an issue on [https://github.com/usnistgov/nfflr/issues](https://github.com/usnistgov/nfflr/issues) or email to brian.decost@nist.gov.

## Code of conduct

Please see our [Code of conduct](https://github.com/usnistgov/jarvis/blob/master/CODE_OF_CONDUCT.md).
