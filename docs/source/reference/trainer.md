# trainer

[nfflr.train](#nfflr.train) provides an [ignite](#ignite)-based trainer utility (and helper functions) for training force fields and general atomistic models.
The trainer supports multi-GPU data-parallel distributed training based on [ignite.distributed](#ignite.distributed).

## Configuration

```{eval-rst}
.. module:: nfflr.train
.. currentmodule:: nfflr.train
.. autosummary::
   :toctree: generated
   :template: classtemplate.rst

   TrainingConfig
```

## [ignite](#ignite)-based trainer

```{eval-rst}
.. module:: nfflr.train
.. currentmodule:: nfflr.train
.. autosummary::
   :toctree: generated

   train
   lr
```

## training setup

```{eval-rst}
.. module:: nfflr.train.trainer
.. currentmodule:: nfflr.train.trainer
.. autosummary::
   :toctree: generated

   get_dataflow
   setup_model_and_optimizer
   setup_trainer
   setup_checkpointing
   setup_evaluators
```
