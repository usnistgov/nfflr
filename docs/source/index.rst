.. NFFLr documentation master file, created by
   sphinx-quickstart on Fri Jun  2 10:22:15 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


NFFLr documentation
===================

Neural Force Field Learning toolkit (NFFLr) is an experimental library intended to support rapid iteration in research on machine learning for atomistic systems.

The initial codebase is a fork of `ALIGNN <https://github.com/usnistgov/alignn>`_, with modified configuration and modeling interfaces to enable usability and performance improvements.

Core concepts
-------------

atomistic systems
^^^^^^^^^^^^^^^^^

A general representation that is easy to work with,
supports efficient batching,
and can be easily converted to input formats of various machine learning models.

.. code-block:: python

    Atoms:
        cell
        positions
        numbers

modeling interface
^^^^^^^^^^^^^^^^^^

Models should have a consistent interface, regardless of the backend!

.. code-block:: python

    NFFLrModel:
        forward(x: Atoms)



training utilities
^^^^^^^^^^^^^^^^^^

High-level training utilities that can be used regardless of the model backend.

`AtomsDataset` and conversion/collation utilities targeting different backends.



.. toctree::
   :maxdepth: 4
   :glob:

   quickstart
   overview
   atoms
   examples



Correspondence
--------------

Please report bugs by `filing an issue on GitHub <https://github.com/usnistgov/nfflr/issues>`_ or email to brian.decost@nist.gov.

Funding support
---------------

NIST-MGI `<https://www.nist.gov/mgi>`_.

Code of conduct
---------------

Please see our `Code of conduct <https://github.com/usnistgov/jarvis/blob/master/CODE_OF_CONDUCT.md>`_


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
