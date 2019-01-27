#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2019 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)

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
