
pyunicorn
=========

.. image:: https://travis-ci.org/pik-copan/pyunicorn.svg?branch=master
    :target: https://travis-ci.org/pik-copan/pyunicorn
.. image:: https://codecov.io/gh/pik-copan/pyunicorn/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/pik-copan/pyunicorn

``pyunicorn`` (**Uni**\ fied **Co**\ mplex Network and **R**\ ecurre\ **N**\ ce
analysis toolbox) is a fully object-oriented Python package for the advanced
analysis and modeling of complex networks. Above the standard measures of
complex network theory such as degree, betweenness and clustering coefficient
it provides some **uncommon but interesting statistics** like Newman's random
walk betweenness. ``pyunicorn`` features novel **node-weighted (node splitting
invariant)** network statistics as well as measures designed for analyzing
**networks of interacting/interdependent networks**.

Moreover, ``pyunicorn`` allows to easily **construct networks from uni- and
multivariate time series and event data** (functional (climate) networks and
recurrence networks). This involves linear and nonlinear measures of time
series analysis for constructing functional networks from multivariate data
(e.g. Pearson correlation, mutual information, event synchronization and event
coincidence analysis). ``pyunicorn`` also features modern techniques of
nonlinear analysis of single and pairs of time series such as recurrence
quantification analysis (RQA), recurrence network analysis and visibility
graphs.


Reference
---------
**Please acknowledge and cite the use of this software and its authors when
results are used in publications or published elsewhere. You can use the
following reference:**

    J.F. Donges, J. Heitzig, B. Beronov, M. Wiedermann, J. Runge, Q.-Y. Feng,
    L. Tupikina, V. Stolbova, R.V. Donner, N. Marwan, H.A. Dijkstra, and J.
    Kurths,
    **Unified functional network and nonlinear time series analysis for complex
    systems science: The pyunicorn package**,
    `Chaos 25, 113101 (2015), doi:10.1063/1.4934554,
    <http://dx.doi.org/10.1063/1.4934554>`_
    `Preprint: arxiv.org:1507.01571 [physics.data-an].
    <http://arxiv.org/abs/1507.01571>`_


Funding
-------
The development of ``pyunicorn`` has been supported by various funding sources,
notably the `German Federal Ministry for Education and Research
<https://www.bmbf.de/en/index.html>`_ (projects `GOTHAM
<http://belmont-gotham.org/>`_ and `CoSy-CC2 <http://cosy.pik-potsdam.de/>`_),
the `Leibniz Association <https://www.leibniz-gemeinschaft.de/en/home/>`_
(projects `ECONS <http://econs.pik-potsdam.de/>`_ and `DominoES
<https://www.pik-potsdam.de/research/projects/activities/dominoes>`_), the
`German National Academic Foundation <https://www.studienstiftung.de/en/>`_,
and the `Stordalen Foundation <http://www.stordalenfoundation.no/>`_ via the
`Planetary Boundary Research Network <http://www.pb-net.org>`_ (PB.net) among
others.


License
-------
``pyunicorn`` is `BSD-licensed <LICENSE.txt>`_ (3 clause).


Code
----
`Stable releases <https://github.com/pik-copan/pyunicorn/releases>`_,
`Development version <https://github.com/pik-copan/pyunicorn>`_

`Changelog <docs/source/changelog.rst>`_, `Contributions <CONTRIBUTIONS.rst>`_


Documentation
-------------
For extensive HTML documentation, jump right to the `pyunicorn homepage
<http://www.pik-potsdam.de/~donges/pyunicorn/>`_. Recent `PDF versions
<http://www.pik-potsdam.de/~donges/pyunicorn/docs/>`_ are also available.

On a local development version, HTML and PDF documentation can be generated
using ``Sphinx``::

    $> pip install --user .[docs]
    $> cd docs; make clean html latexpdf


Dependencies
------------
``pyunicorn`` is implemented in Python 3. The software is written and tested on
Linux and MacOSX, but it is also in active use on Windows. ``pyunicorn`` relies
on the following open source or freely available packages, which need to be
installed on your machine. For exact dependency information, see ``setup.cfg``.

Required at runtime:
  - `Numpy <http://www.numpy.org/>`_
  - `Scipy <http://www.scipy.org/>`_
  - `python-igraph <http://igraph.org/>`_

Optional *(used only in certain classes and methods)*:
  - `PyNGL <http://www.pyngl.ucar.edu/Download/>`_
    (for ``NetCDFDictionary``)
  - `netcdf4-python <http://unidata.github.io/netcdf4-python/>`_
    (for ``Data`` and ``NetCDFDictionary``)
  - `Matplotlib <http://matplotlib.org/>`_
  - `Matplotlib Basemap Toolkit <http://matplotlib.org/basemap/>`_
    (for drawing maps)
  - `mpi4py <https://bitbucket.org/mpi4py/mpi4py>`_
    (for parallelizing costly computations)
  - `Sphinx <http://sphinx-doc.org/>`_
    (for generating documentation)
  
To install these dependencies, please follow the instructions for your system's
package manager or consult the libraries' homepages. An easy way to go may be a
Python distribution like `Anaconda <https://www.anaconda.com/distribution/>`_
that already includes many libraries.


Installation
------------
Before installing ``pyunicorn`` itself, we recommend to make sure that the
required dependencies are installed using your preferred installation method for
Python libraries. Afterwards, the package can be installed in the standard way
from the Python Package Index (PyPI).

**Linux, MacOSX**

With the ``pip`` package manager::

    $> pip install pyunicorn
        
On Fedora OS, use::

    $> dnf install python3-pyunicorn

**Windows**

Install the latest version of the `Microsoft C++ Build Tools
<https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_, and then::

    $> pip install pyunicorn

**Development version**

To use a newer version of ``pyunicorn`` than the latest official release on
PyPI, download the source code from the Github repository and, instead of the
above, execute::

    $> pip install -e .


Test suite
----------
Before committing changes or opening a pull request (PR) to the code base,
please make sure that all tests pass. The test suite is managed by `tox
<http://tox.readthedocs.io/>`_ and configured to use system-wide packages when
available. Install the test dependencies as follows::

    $> pip install .[testing]

The test suite can be run from anywhere in the project tree by issuing::

    $> tox

To display the defined test environments and target them individually::

    $> tox -l
    $> tox -e units,pylint,docs

To test individual files::

    $> pytest           tests/test_core/TestNetwork.py   # unit tests
    $> pytest --flake8  pyunicorn/core/network.py        # style
    $> pylint           pyunicorn/core/network.py        # static code analysis


Mailing list
------------
Not implemented yet.
