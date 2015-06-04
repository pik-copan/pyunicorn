
Changelog
=========

A summary of major changes made in each release of ``pyunicorn``:

**0.4.2**
 - Improved Pylint compliance.

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
