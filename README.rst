
pyunicorn
=========

``pyunicorn`` (**Uni**\ fied **Co**\ mplex Network and **R**\ ecurre\ **N**\ ce
analysis toolbox) is a fully object-oriented Python package for the advanced
analysis and modeling of complex networks. Above the standard measures of
complex network theory such as degree, betweenness and clustering coefficient
it provides some **uncommon but interesting statistics** like Newman's random
walk betweenness. ``pyunicorn`` features novel **node-weighted (node splitting
invariant)** network statistics as well as measures designed for analyzing
**networks of interacting/interdependent networks**.

Moreover, ``pyunicorn`` allows to easily **construct networks from uni- and
multivariate time series data** (functional (climate) networks and recurrence
networks). This involves linear and nonlinear measures of time series analysis
for constructing functional networks from multivariate data as well as modern
techniques of nonlinear analysis of single time series like recurrence
quantification analysis (RQA) and recurrence network analysis.

Get more information
--------------------
This page only contains bare-bones instructions for installing ``pyunicorn``.
For more information, jump right to the `pyunicorn homepage`_.

.. _pyunicorn homepage: http://www.pik-potsdam.de/~donges/pyunicorn/

Get the latest code
-------------------
Git repository not yet public.

Dependencies
------------
``pyunicorn`` relies on the following additional open source or freely
available packages that have to be installed on your machine.

Required:
  - `Numpy <http://numpy.scipy.org/>`_ 1.8+
  - `Scipy <http://www.scipy.org/>`_ 0.14+
  - `Weave <https://github.com/scipy/weave>`_ 0.15+
  - `igraph, python-igraph <http://igraph.sourceforge.net/>`_ 0.7+

Optional *(required for certain classes and methods)*:
  - `PyNGL <http://www.pyngl.ucar.edu/Download/>`_ (for class
    NetCDFDictionary)
  - `netcdf4-python <http://code.google.com/p/netcdf4-python/>`_ (for classes
    Data and NetCDFDictionary)
  - `Matplotlib <http://matplotlib.sourceforge.net>`_ 1.3+
  - `Matplotlib Basemap Toolkit <http://matplotlib.org/basemap/>`_ (for drawing
    maps)
  - `mpi4py <http://code.google.com/p/mpi4py/>`_ (for parallelizing costly
    computations)
  - `Sphinx <http://sphinx-doc.org/>`_ (for generating documentation in
    formats other than HTML, e.g. PDF)

Install pyunicorn
-----------------
Install ``pyunicorn`` using Python's ``distutils`` with the supplied
``setup.py`` script::

   $> python setup.py install

Depending on your system, you may need root priviledges.  On UNIX-based
operating systems (Linux, MacOSX etc.) this is achieved with ``sudo``.

Install the latest stable release of ``pyunicorn`` via the Python package
index::

   $> pip install pyunicorn

Documentation
-------------
Using ``Sphinx``, currently HTML and PDF can be generated::

    $> cd docs; make clean html latexpdf

Mailing list
------------
Not implemented yet.

Test suite
----------
Before committing changes to the codebase, please make sure that all tests
pass. The test suite is managed by `tox <https://testrun.org/tox/>`_ and
configured to use system-wide packages when available. Thus to avoid frequent
waiting, we recommend you to install the current versions of the following
packages::

    $> pip install tox nose networkx Sphinx
    $> pip install pylint pytest pytest-xdist pytest-flakes pytest-pep8

The test suite can be run from anywhere in the project tree by issuing::

    $> tox

To expose the defined test environments and target them independently::

    $> tox -l
    $> tox -e py27-nose,py27-style

To test single files::

    $> tests/test_doctests.py core.network      # doctests
    $> nosetests -vs tests/core/TestNetwork.py  # unit tests
    $> py.test pyunicorn/core/network.py        # style

License
-------
``pyunicorn`` is **BSD-licensed** (3 clause), see ``LICENSE.txt``.
