#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2019 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)
#
# Please acknowledge and cite the use of this software and its authors
# when results are used in publications or published elsewhere.
#
# You can use the following reference:
# J.F. Donges, J. Heitzig, B. Beronov, M. Wiedermann, J. Runge, Q.-Y. Feng,
# L. Tupikina, V. Stolbova, R.V. Donner, N. Marwan, H.A. Dijkstra,
# and J. Kurths, "Unified functional network and nonlinear time series analysis
# for complex systems science: The pyunicorn package"

"""
core
====

Provides classes for analyzing spatially embedded complex networks, handling
multivariate data and generating time series surrogates.

Related Publications
~~~~~~~~~~~~~~~~~~~~
[Donges2011a]_, [Heitzig2012]_, [Donges2012]_

To do
~~~~~
  - A lot - See current product backlog.
  - Clean up MapPlots class -> Alex!?

Known Bugs
~~~~~~~~~~
  - ...

"""

#
#  Import classes
#

from .network import Network, NetworkError, nz_coords, cached_const
from .geo_network import GeoNetwork
from .grid import Grid
from .data import Data
from .interacting_networks import InteractingNetworks
from .netcdf_dictionary import NetCDFDictionary
from .resistive_network import ResNetwork

#
#  Set global constants
#

#  Mean earth radius in kilometers
EARTH_RADIUS = 6367.5
"""(float) - The earth's mean radius in kilometers."""
