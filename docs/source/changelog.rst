
Changelog
=========

A summary of major changes made in each release of ``pyunicorn``:

0.7.0
-----

- fixed some bugs
- improved test coverage
- established Windows installation,
  established CI for Python 3.8 - 3.12 on Linux,
  established CI for Python 3.11 on macOS and Windows,
  discontinued Python 3.7 support
- worked towards Cythonizing all C extensions, few remaining
- resolved type conversion and indexation problems at Python/Cython interface
- shifted internal index in ``RecurrencePlot`` line distribution methods
- replaced outdated ``utils/progressbar`` with ``tqdm`` dependency
- added ``multiprocess`` dependency
- reviewed some n.s.i. measures in ``core.Network``,
  enabled their ``corrected`` mode
- reviewed tutorials, added tutorial on ``CoupledClimateNetworks``
  and directly included tutorial-notebooks in documentation
- replaced old ``climate.MapPlots`` class with simplified ``climate.MapPlot``,
  using ``cartopy`` optional dependency
- added classes ``core.SpatialNetwork``, ``eventseries.EventSeries``, ``climate.EventSeriesClimateNetwork``
- added methods ``cross_degree_density()``, ``local_efficiency()``, ``global_efficiency()``,
  ``average_cross_closeness()`` and ``total_cross_degree()`` to ``core.InteractingNetworks``
- added timeseries entropy measures to ``timeseries.RecurrencePlot``
- added ``cross_link_distance()`` to ``CoupledClimateNetwork``
- reviewed ``core.Grid``
- reviewed memoization/caching

0.6.1
-----
 - Fixed some bugs and compatibility issues.
 - Improved test framework.
 - Added pyunicorn description paper reference to all code files.

0.6.0
-----
 - Migrated from Python 2.7 to Python 3.7.
 - Completed transition from ``Weave`` to ``Cython``.
 - Added Event Coincidence Analysis.

0.5.2
-----
 - Updated test suite and ``Travis CI``.

0.5.1
-----
 - Added reference to pyunicorn description paper published in the
   journal Chaos.

0.5.0
-----
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

0.4.1
-----
 - Removed a whole lot of ``get_`` s from the API. For example,
   ``Network.get_degree()`` is now ``Network.degree()``.
 - Fixed some minor bugs.

0.4.0
-----
 - Restructured package (subpackages: ``core``, ``climate``, ``timeseries``,
   ``funcnet``, ``utils``).
 - Removed dependencies: ``Pysparse``, ``PyNio``, ``progressbar``.
 - Added a module for resistive networks.
 - Switched to ``tox`` for test suite management.
 - Ensured PEP8 and PyFlakes compliance.

0.3.2
-----
 - Fixed some minor bugs.
 - Switched to ``Sphinx`` documentation system.

0.3.1
-----
 - First public release of ``pyunicorn``.
