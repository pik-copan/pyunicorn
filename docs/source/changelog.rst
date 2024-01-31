
Changelog
=========

A summary of major changes made in each release of ``pyunicorn``:

0.7.0
-----

- fixed some bugs and improved test coverage 
  (see `Release Milestone <https://github.com/pik-copan/pyunicorn/milestone/1?closed=1>`_ for all resolved issues)

- established Windows installation 
  (`#160 <https://github.com/pik-copan/pyunicorn/issues/160>`_,
  `#127 <https://github.com/pik-copan/pyunicorn/issues/127>`_)
- established CI for Python 3.8 - 3.12 on Linux
  (`#191 <https://github.com/pik-copan/pyunicorn/issues/191>`_,
  `#192 <https://github.com/pik-copan/pyunicorn/pull/192>`_),
- established CI for Python 3.11 on macOS and Windows
  (`#201 <https://github.com/pik-copan/pyunicorn/issues/201>`_)
- discontinued Python 3.7 support
  (`4cf6969 <https://github.com/pik-copan/pyunicorn/commit/4cf6969c40de39f01f31ea141767ec67cc3d6d31>`_)

- replaced old ``climate.MapPlots`` class with simplified ``climate.MapPlot``,
  using ``cartopy`` optional dependency
  (`#174 <https://github.com/pik-copan/pyunicorn/pull/202>`_,
  `#203 <https://github.com/pik-copan/pyunicorn/issues/203>`_)
- added class ``core.SpatialNetwork``;
  reviewed ``core.Grid``;
  added Watts-Strogatz Model to ``core.Network``;
  added ``cross_link_distance()`` to ``CoupledClimateNetwork``;
  added timeseries entropy measures to ``timeseries.RecurrencePlot``;
  added methods ``cross_degree_density()``, ``local_efficiency()``, ``global_efficiency()``,
  ``average_cross_closeness()`` and ``total_cross_degree()`` to ``core.InteractingNetworks`` 
  (`#131 <https://github.com/pik-copan/pyunicorn/pull/131>`_)
- added classes ``eventseries.EventSeries`` and ``climate.EventSeriesClimateNetwork``
  (`#156 <https://github.com/pik-copan/pyunicorn/pull/156>`_)

- worked towards Cythonizing all C extensions, few remaining
  (`#128 <https://github.com/pik-copan/pyunicorn/issues/128>`_,
  `#142 <https://github.com/pik-copan/pyunicorn/issues/142>`_,
  `#145 <https://github.com/pik-copan/pyunicorn/issues/145>`_,
  `#187 <https://github.com/pik-copan/pyunicorn/issues/187>`_,
  `#195 <https://github.com/pik-copan/pyunicorn/pull/195>`_)
- resolved type conversion and indexation problems at Python/Cython interface
  (`#126 <https://github.com/pik-copan/pyunicorn/issues/126>`_,
  `#141 <https://github.com/pik-copan/pyunicorn/issues/141>`_,
  `#145 <https://github.com/pik-copan/pyunicorn/issues/145>`_,
  `#162 <https://github.com/pik-copan/pyunicorn/issues/162>`_,
  `#163 <https://github.com/pik-copan/pyunicorn/issues/163>`_,
  `#201 <https://github.com/pik-copan/pyunicorn/issues/201>`_)
- shifted internal index in ``RecurrencePlot`` line distribution methods
  (`#166 <https://github.com/pik-copan/pyunicorn/issues/166>`_,
  `#209 <https://github.com/pik-copan/pyunicorn/pull/209>`_)

- replaced outdated ``utils/progressbar`` with ``tqdm`` dependency
  (`#202 <https://github.com/pik-copan/pyunicorn/pull/202>`_)
- replaced optional dependecy ``netcdf4`` with ``h5netcdf``
  (`cd8ee00 <https://github.com/pik-copan/pyunicorn/commit/cd8ee00a534c0eae9440414d38a0eaaa5100aaec>`_,
  also see `#12 <https://github.com/pik-copan/pyunicorn/issues/12>`_,
  `#210 <https://github.com/pik-copan/pyunicorn/issues/210>`_,)
- added ``multiprocess`` dependency to enable parallelization
  (`#142 <https://github.com/pik-copan/pyunicorn/issues/142>`_)

- reviewed some n.s.i. measures in ``core.Network``, enabled their ``corrected`` mode
  (`#153 <https://github.com/pik-copan/pyunicorn/pull/153>`_)
- reviewed tutorials, added tutorial on ``CoupledClimateNetworks``
  and directly included tutorial-notebooks in documentation
  (`#175 <https://github.com/pik-copan/pyunicorn/pull/175>`_,
  `#180 <https://github.com/pik-copan/pyunicorn/pull/180>`_,
  `#185 <https://github.com/pik-copan/pyunicorn/issues/185>`_,
  `#190 <https://github.com/pik-copan/pyunicorn/pull/190>`_,
  `#213 <https://github.com/pik-copan/pyunicorn/pull/213>`_
  )
- reviewed memoization/caching
  (`#124 <https://github.com/pik-copan/pyunicorn/issues/124>`_,
  `#148 <https://github.com/pik-copan/pyunicorn/issues/148>`_)

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
