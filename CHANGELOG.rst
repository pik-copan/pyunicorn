
Changelog
=========

0.8.0
-----

Package:

- Improved `test coverage <https://app.codecov.io/gh/pik-copan/pyunicorn?search=&trend=all%20time>`_.
- Improved style and linting by revising and reducing global message disables
  (`#224 <https://github.com/pik-copan/pyunicorn/pull/224>`_,
  `#231 <https://github.com/pik-copan/pyunicorn/pull/231>`_,
  `#233 <https://github.com/pik-copan/pyunicorn/pull/233>`_).
- Ensured ``numpy=2.0`` support
  (`#234 <https://github.com/pik-copan/pyunicorn/pull/234>`_).
- Removed experimental and untested code
  (`#40 <https://github.com/pik-copan/pyunicorn/issues/40>`_,
  `#239 <https://github.com/pik-copan/pyunicorn/pull/239>`_).

New/Updated functionality:

- Extended Caching to ``RecurrenceNetwork`` and child classes as well as ``Surrogates``.
  (`#236 <https://github.com/pik-copan/pyunicorn/pull/236>`_).
- Revised API of ``Surrogates`` to conform to OO structure assumed by ``Cached``.
  (`#236 <https://github.com/pik-copan/pyunicorn/pull/236>`_).

Documentation:

- Removed malfunctioning links from Introduction
  (`369b086 <https://github.com/pik-copan/pyunicorn/commit/369b086a8535dbfad6651caa62bee5a5488a0bfd>`_).
- Moved tutorial notebooks into ``docs``, dropped ``nbsphinx-link`` dependency
  (`c54102e <https://github.com/pik-copan/pyunicorn/commit/c54102e42b767271be6299f8bf8170b27bda28e6>`_).
- Added ``EventSeriesClimateNetwork`` to API documentation
  (`#232 <https://github.com/pik-copan/pyunicorn/pull/232>`_).
- Removed outdated MathJax path for correct math display
  (`0dd133c <https://github.com/pik-copan/pyunicorn/commit/0dd133c59da252b8c0e0e17f82290881508d0274>`_).

Bug fixes:

- Fixed unnoticed bug in ``SpatialNetwork.{in|out}average_link_distance()``
  (`6b40587 <https://github.com/pik-copan/pyunicorn/commit/6b405873bede4ec18cd72164c734ed47964d2930>`_).
- Substituted deprecated shorthand ``scipy.sparse.*_matrix.A`` for ``scipy>=1.14`` compatibility
  (`1d96e58 <https://github.com/pik-copan/pyunicorn/commit/1d96e58040c831afdcd7f7bf97be3ebd6ae6815a>`_).
- Substituted variable name ``I`` in C code, which might interfere with C's own macro for complex numbers
  (`#225 <https://github.com/pik-copan/pyunicorn/issues/225>`_,
  `#226 <https://github.com/pik-copan/pyunicorn/pull/232>`_).
- Fixed setup of Travis-CI on Windows builds
  (`#237 <https://github.com/pik-copan/pyunicorn/pull/237>`_).

0.7.0
-----

Package:

- Migrated to PEP 517/518 package format
  (`a6c4c83 <https://github.com/pik-copan/pyunicorn/commit/a6c4c83905fcc4b73f46643fbe2f160917755e0e>`_).
- Added full Windows support
  (`#159 <https://github.com/pik-copan/pyunicorn/issues/159>`_,
  `#160 <https://github.com/pik-copan/pyunicorn/issues/160>`_).
- Reestablished `CI <https://app.travis-ci.com/github/pik-copan/pyunicorn>`_ on Linux
  (`#191 <https://github.com/pik-copan/pyunicorn/issues/191>`_,
  `#192 <https://github.com/pik-copan/pyunicorn/pull/192>`_)
  and added macOS and Windows
  (`#214 <https://github.com/pik-copan/pyunicorn/pull/214>`_).
- Improved `test coverage <https://app.codecov.io/gh/pik-copan/pyunicorn?search=&trend=all%20time>`_.
- Discontinued Python 3.7 support
  (`4cf6969 <https://github.com/pik-copan/pyunicorn/commit/4cf6969c40de39f01f31ea141767ec67cc3d6d31>`_).
- Replaced optional dependecy ``netcdf4`` with ``h5netcdf``
  (`cd8ee00 <https://github.com/pik-copan/pyunicorn/commit/cd8ee00a534c0eae9440414d38a0eaaa5100aaec>`_,
  `#12 <https://github.com/pik-copan/pyunicorn/issues/12>`_,
  `#210 <https://github.com/pik-copan/pyunicorn/issues/210>`_).
- Replaced outdated ``progressbar`` with ``tqdm``
  (`#202 <https://github.com/pik-copan/pyunicorn/pull/202>`_).

Documentation:

- Added new tutorials
  (`#175 <https://github.com/pik-copan/pyunicorn/pull/175>`_,
  `#180 <https://github.com/pik-copan/pyunicorn/pull/180>`_,
  `#190 <https://github.com/pik-copan/pyunicorn/pull/190>`_).
- Edited tutorials and included notebooks in documentation
  (`#179 <https://github.com/pik-copan/pyunicorn/pull/179>`_,
  `#185 <https://github.com/pik-copan/pyunicorn/issues/185>`_,
  `#213 <https://github.com/pik-copan/pyunicorn/pull/213>`_).

New/Updated functionality:

- Generalized spatial and interacting network analysis
  (`#131 <https://github.com/pik-copan/pyunicorn/pull/131>`_):
  added ``SpatialNetwork`` class, added Watts-Strogatz model to ``Network``,
  added new metrics to ``RecurrencePlot``, ``CoupledClimateNetwork`` and
  ``InteractingNetworks``.
- Added ``EventSeries`` and ``EventSeriesClimateNetwork`` classes
  (`#156 <https://github.com/pik-copan/pyunicorn/pull/156>`_).
- Extended n.s.i. measures in ``Network`` with directed and weighted versions
  (`#153 <https://github.com/pik-copan/pyunicorn/pull/153>`_).
- Replaced ``MapPlots`` class with simplified ``MapPlot`` based on ``Cartopy``
  (`#174 <https://github.com/pik-copan/pyunicorn/pull/174>`_,
  `#203 <https://github.com/pik-copan/pyunicorn/issues/203>`_).

Extensions:

- Overhauled the Python/Cython interface
  (`3dab5bf <https://github.com/pik-copan/pyunicorn/commit/3dab5bf89d2e224fc319ddd64aeeecc480f27fba>`_,
  `402197f <https://github.com/pik-copan/pyunicorn/commit/402197fedff6dc4ce9796b5d2c32bb63ef6ecba8>`_).
- Made Cython/C extensions compatible with MSVC
  (`#160 <https://github.com/pik-copan/pyunicorn/issues/160>`_,
  `#165 <https://github.com/pik-copan/pyunicorn/issues/165>`_).
- Ported most of the remaining C extensions to Cython
  (`#128 <https://github.com/pik-copan/pyunicorn/issues/128>`_,
  `#142 <https://github.com/pik-copan/pyunicorn/issues/142>`_,
  `#145 <https://github.com/pik-copan/pyunicorn/issues/145>`_,
  `#187 <https://github.com/pik-copan/pyunicorn/issues/187>`_,
  `#195 <https://github.com/pik-copan/pyunicorn/pull/195>`_).
  
Bug fixes:

- Resolved indexing and typing problems in extensions
  (`#126 <https://github.com/pik-copan/pyunicorn/issues/126>`_,
  `#141 <https://github.com/pik-copan/pyunicorn/issues/141>`_,
  `#145 <https://github.com/pik-copan/pyunicorn/issues/145>`_,
  `#162 <https://github.com/pik-copan/pyunicorn/issues/162>`_,
  `#163 <https://github.com/pik-copan/pyunicorn/issues/163>`_).
- Overhauled the memoization/caching system
  (`#124 <https://github.com/pik-copan/pyunicorn/issues/124>`_,
  `#148 <https://github.com/pik-copan/pyunicorn/issues/148>`_,
  `#219 <https://github.com/pik-copan/pyunicorn/pull/219>`_).
- Shifted the histogram index in ``RecurrencePlot`` line distributions
  (`#166 <https://github.com/pik-copan/pyunicorn/issues/166>`_,
  `#209 <https://github.com/pik-copan/pyunicorn/pull/209>`_).
- Resolved numerous other issues related to inheritance and method overloading,
  deprecated APIs, etc. For a full list, see the `release milestone
  <https://github.com/pik-copan/pyunicorn/milestone/1?closed=1>`_.

0.6.1
-----
- Fixed some bugs and compatibility issues.
- Improved test framework.
- Added ``pyunicorn`` description paper reference to all code files.

0.6.0
-----
- Migrated from Python 2.7 to Python 3.7.
- Completed transition from ``Weave`` to ``Cython``.
- Added Event Coincidence Analysis.

0.5.2
-----
- Updated test suite and CI.

0.5.1
-----
- Added reference to ``pyunicorn`` description paper published in the
  journal *Chaos*.

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
- Set up continuous testing with Travis-CI.
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
