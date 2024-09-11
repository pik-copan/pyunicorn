=========
pyunicorn
=========

.. image:: https://app.travis-ci.com/pik-copan/pyunicorn.svg?branch=master
  :target: https://app.travis-ci.com/github/pik-copan/pyunicorn

.. image:: https://codecov.io/gh/pik-copan/pyunicorn/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/pik-copan/pyunicorn

.. image:: https://img.shields.io/pypi/v/pyunicorn
  :target: https://pypi.org/project/pyunicorn/

.. image:: https://img.shields.io/pypi/pyversions/pyunicorn
  :target: https://pypi.org/project/pyunicorn/

.. image:: https://zenodo.org/badge/33720178.svg
  :target: https://zenodo.org/badge/latestdoi/33720178

About
=====
``pyunicorn`` (**Uni**\ fied **Co**\ mplex Network and **R**\ ecurre\ **N**\ ce
analysis toolbox) is an object-oriented Python package for the advanced analysis
and modeling of complex networks. Beyond the standard **measures of complex
network theory** (such as *degree*, *betweenness* and *clustering coefficients*), it
provides some uncommon but interesting statistics like *Newman's random walk
betweenness*. ``pyunicorn`` also provides novel **node-weighted** *(node splitting invariant)*
network statistics, measures for analyzing networks of **interacting/interdependent
networks**, and special tools to model **spatially embedded** complex networks.

Moreover, ``pyunicorn`` allows one to easily *construct networks* from uni- and
multivariate time series and event data (**functional/climate networks** and
**recurrence networks**). This involves linear and nonlinear measures of
**time series analysis** for constructing functional networks from multivariate data
(e.g., *Pearson correlation*, *mutual information*, *event synchronization* and *event
coincidence analysis*). ``pyunicorn`` also features modern techniques of
nonlinear analysis of time series (or pairs thereof), such as *recurrence
quantification analysis* (RQA), *recurrence network analysis* and *visibility
graphs*.

``pyunicorn`` is **fast**, because all costly computations are performed in
compiled C code. It can handle **large networks** through the
use of sparse data structures. The package can be used interactively, from any
Python script, and even for parallel computations on large cluster architectures.
For information about individual releases,
see our `CHANGELOG <CHANGELOG.rst>`_ and `CONTRIBUTIONS <CONTRIBUTIONS.rst>`_.


License
-------
``pyunicorn`` is `BSD-licensed <LICENSE.txt>`_ (3 clause).

Reference
---------
*Please acknowledge and cite the use of this software and its authors when
results are used in publications or published elsewhere. You can use the
following reference:*

    J.F. Donges, J. Heitzig, B. Beronov, M. Wiedermann, J. Runge, Q.-Y. Feng,
    L. Tupikina, V. Stolbova, R.V. Donner, N. Marwan, H.A. Dijkstra, and J.
    Kurths.
    **"Unified functional network and nonlinear time series analysis for complex
    systems science: The pyunicorn package"**.
    Chaos 25, 113101 (2015), `doi:10.1063/1.4934554
    <http://dx.doi.org/10.1063/1.4934554>`_, Preprint: `arxiv.org:1507.01571
    <http://arxiv.org/abs/1507.01571>`_ [physics.data-an].

Funding
-------
The development of ``pyunicorn`` has been supported by various funding sources,
notably the `German Federal Ministry for Education and Research
<https://www.bmbf.de/bmbf/en/home/home_node.html>`_ (projects `GOTHAM
<https://www.belmontforum.org/projects>`_ and `CoSy-CC2
<http://cosy.pik-potsdam.de/>`_), the `Leibniz Association
<https://www.leibniz-gemeinschaft.de/en/>`_ (projects `ECONS
<http://econs.pik-potsdam.de/>`_ and `DominoES
<https://www.pik-potsdam.de/en/institute/departments/activities/dominoes>`_),
the `German National Academic Foundation <https://www.studienstiftung.de/en/>`_,
and the `Stordalen Foundation <http://www.stordalenfoundation.no/>`_ via the
`Planetary Boundary Research Network
<https://web.archive.org/web/20200212214011/http://pb-net.org/>`_ (PB.net) among
others.

Getting Started
===============

Installation
------------
Official releases
.................
`Stable releases <https://pypi.org/project/pyunicorn/#history>`_ can be
installed directly from the `Python Package Index (PyPI)
<https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-from-pypi>`_::

    $> pip install pyunicorn

Alternatively, source distributions can be downloaded from the
`GitHub Releases <https://github.com/pik-copan/pyunicorn/releases>`_.

On **Windows**, please *first* install the latest version of the `Microsoft C++ Build
Tools <https://wiki.python.org/moin/WindowsCompilers>`_, which is required for
compiling Cython modules.

Current development version
...........................
In order to use a `newer version <https://github.com/pik-copan/pyunicorn>`_,
please follow the ``pip`` instructions for installing from `version control
<https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-from-vcs>`_
or from a `local source tree
<https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-from-a-local-src-tree>`_.

Dependencies
............
``pyunicorn`` is implemented in `Python 3 <https://docs.python.org/3/>`_ /
`Cython 3 <https://cython.org/>`_, is `tested
<https://app.travis-ci.com/github/pik-copan/pyunicorn>`_ on *Linux*, *macOS*
and *Windows*, and relies on the following packages:

- Required:

  - `numpy <http://www.numpy.org/>`_, `scipy <http://www.scipy.org/>`_
  - `python-igraph <http://igraph.org/>`_ (for ``Network``)
  - `h5netcdf <https://h5netcdf.org/>`_ (for ``Data``, ``NetCDFDictionary``)
  - `tqdm <https://tqdm.github.io/>`_ (for progress bars)

- Optional:

  - `Matplotlib <http://matplotlib.org/>`_,
    `Cartopy <https://scitools.org.uk/cartopy/docs/latest/index.html>`_
    (for plotting features)
  - `mpi4py <https://github.com/mpi4py/mpi4py>`_
    (for parallelizing costly computations)
  - `Sphinx <http://sphinx-doc.org/>`_
    (for generating documentation)
  - `Jupyter Notebook <https://jupyter-notebook.readthedocs.io/en/latest/>`_
    (for tutorial notebooks)


Documentation
-------------
For extensive HTML documentation, jump right to the `homepage
<http://www.pik-potsdam.de/~donges/pyunicorn/>`_. In a local source tree,
HTML and PDF documentation can be generated using ``Sphinx``::

    $> pip install .[docs]
    $> cd docs; make clean html latexpdf

Tuturials
---------

For some example applications look into the
`tutorials <docs/source/examples/tutorials/>`_ provided with the documentation.
They are designed to be self-explanatory, and are set up as Jupyter notebooks.

Development
===========

Test suite
----------
Before committing changes or opening a pull request (PR) to the code base,
please make sure that all tests pass. The test suite is managed by `tox
<https://tox.wiki/>`_ and is configured to use system-wide packages
when available. Install the test dependencies as follows::

    $> pip install -e .[tests]

The test suite can be run from anywhere in the project tree by issuing::

    $> tox

To display the defined test environments and target them individually::

    $> tox -l
    $> tox -e style,lint,test,docs

To test individual files::

    $> flake8 src/pyunicorn/core/network.py     # style check
    $> pylint src/pyunicorn/core/network.py     # static code analysis
    $> pytest tests/test_core/test_network.py   # unit tests
