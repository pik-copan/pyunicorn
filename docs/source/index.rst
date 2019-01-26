
Introduction
============

.. include:: ../../README.rst
    :start-after: =========
    :end-before: Reference

For example, to generate a recurrence network with 1000 nodes from a sinusoidal
signal and compute its network transitivity you simply need to type

.. literalinclude:: ../../examples/modules/timeseries/recurrence_network.py

The package provides special tools to analyze and model **spatially embedded**
complex networks.

``pyunicorn`` is **fast** because all costly computations are performed in
compiled C, C++ and Fortran code. It can handle **large networks** through the
use of sparse data structures. The package can be used interactively, from any
Python script and even for parallel computations on large cluster
architectures.
