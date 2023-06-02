# NFFLr

Neural Force Field Learning toolkit (NFFLr) is an experimental library intended to support rapid iteration in research on machine learning for atomistic systems.

The initial codebase is a fork of [ALIGNN](https://github.com/usnistgov/alignn), with modified configuration and modeling interfaces to enable usability and performance improvements.


## atomistic systems
A general representation that is easy to work with,
supports efficient batching,
and can be easily converted to input formats of various machine learning models.

```
Atoms:
    cell
    positions
    numbers
```

## modeling interface
Models should have a consistent interface, regardless of the backend!

```
NFFLrModel:
    forward(x: Atoms)
```


## training utilities
High-level training utilities that can be used regardless of the model backend.

`AtomsDataset` and conversion/collation utilities targeting different backends.


# Correspondence

Please report bugs as Github issues (https://github.com/usnistgov/nfflr/issues) or email to brian.decost@nist.gov.

# Funding support

NIST-MGI (https://www.nist.gov/mgi).

# Code of conduct

Please see [Code of conduct](https://github.com/usnistgov/jarvis/blob/master/CODE_OF_CONDUCT.md)
