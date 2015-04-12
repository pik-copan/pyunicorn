
API
===

:Release: |release|
:Date: |today|

``pyunicorn`` consists of five subpackages, where the ``core`` and
``utils.mpi`` namespaces are to be accessed by calling ``import pyunicorn``.
All subpackages except for ``utils`` directly export the classes defined in
their submodules.

core
----
General network analysis and modeling.

.. toctree::
    :maxdepth: 1
    :glob:

    api/core/*

climate
-------
Constructing and analysing climate networks, related climate data analysis.

.. toctree::
    :maxdepth: 1
    :glob:

    api/climate/*

timeseries
----------
Recurrence plots, recurrence networks, multivariate extensions and visibility
graph analysis of time series. Time series surrogates for significance testing.

.. toctree::
    :maxdepth: 1
    :glob:

    api/timeseries/*

funcnet
-------
Constructing and analysing general functional networks.

.. toctree::
    :maxdepth: 1
    :glob:

    api/funcnet/*

utils
-----
Parallelization, interactive network navigator, helpers.

.. toctree::
    :maxdepth: 1
    :glob:

    api/utils/*
