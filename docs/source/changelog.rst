
Changelog
=========

A summary of major changes made in each release of ``pyunicorn``:

**0.6.1**
 - Fixed some bugs and compatibility issues.
 - Improved test framework.
 - Added pyunicorn description paper reference to all code files.

**0.6.0**
 - Migrated from Python 2.7 to Python 3.7.
 - Completed transition from ``Weave`` to ``Cython``.
 - Added Event Coincidence Analysis.

**0.5.2**
 - Updated test suite and ``Travis CI``.

**0.5.1**
 - Added reference to pyunicorn description paper published in the
   journal Chaos.

**0.5.0**
 - Substantial update of ``CouplingAnalysis``.
 - New methods in ``RecurrenceNetwork``: ``transitivity_dim_single_scale``,
   ``local_clustering_dim_single_scale``.
 - Renamed time-directed measures in ``VisibilityGraph``: ``left/right`` ->
   ``retarded/advanced``.
 - Improved documentation and extended publication list.
 - Began transition from ``Weave`` to ``Cython``.
 - Added unit tests and improved Pylint compliance.
 - Set up continuous testing with Travis CI.
 - Fixed some minor bugs.

**0.4.1**
 - Removed a whole lot of ``get_`` s from the API. For example,
   ``Network.get_degree()`` is now ``Network.degree()``.
 - Fixed some minor bugs.

**0.4.0**
 - Restructured package (subpackages: ``core``, ``climate``, ``timeseries``,
   ``funcnet``, ``utils``).
 - Removed dependencies: ``Pysparse``, ``PyNio``, ``progressbar``.
 - Added a module for resistive networks.
 - Switched to ``tox`` for test suite management.
 - Ensured PEP8 and PyFlakes compliance.

**0.3.2**
 - Fixed some minor bugs.
 - Switched to ``Sphinx`` documentation system.

**0.3.1**
 - First public release of ``pyunicorn``.
