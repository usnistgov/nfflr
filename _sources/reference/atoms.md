# nfflr.Atoms

This document describes the top-level nfflr API.

```{eval-rst}
.. module:: nfflr
.. currentmodule:: nfflr
.. autoclass:: nfflr.Atoms

```

## Graph construction

```{eval-rst}
.. module:: nfflr.nn
.. currentmodule:: nfflr.nn
```

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: newclass.rst

   PeriodicRadiusGraph
   PeriodicKShellGraph
   PeriodicAdaptiveRadiusGraph
```

## atom representations

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: newclass.rst

   AtomType
   AtomPairType
   AttributeEmbedding
   AtomicNumberEmbedding
   PeriodicTableEmbedding

```

## batching

```{eval-rst}
.. currentmodule:: nfflr
.. autosummary::
   :toctree: generated
   :nosignatures:

   batch
   unbatch
```


## conversion to other formats

```{eval-rst}
.. currentmodule:: nfflr

.. autosummary::
   :toctree: generated
   :nosignatures:

   spglib_cell
   to_ase
```
