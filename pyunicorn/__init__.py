# This file is part of pyunicorn
# (Unified Complex Network and Recurrence Analysis Toolbox).
#
# Copyright (C) 2008--2019 Jonathan F. Donges and pyunicorn authors
#
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)

"""
pyunicorn
=========


Subpackages
-----------
core
    Spatially embedded complex networks and multivariate data
climate
    Climate networks
funcnet
    Functional networks
timeseries
    Time series surrogates

To Do
-----
   - A lot - See current product backlog.
   - Clean up MapPlots class -> Alex!?

"""

from pkg_resources import get_distribution

from .utils import mpi
from .core import *


__author__ = "Jonathan F. Donges <donges@pik-potsdam.de>"
__copyright__ = \
    "Copyright (C) 2008-2019 Jonathan F. Donges and pyunicorn authors"
__license__ = "BSD (3-clause)"
__url__ = "http://www.pik-potsdam.de/members/donges/software"
__version__ = get_distribution(__package__).version
__docformat__ = "restructuredtext en"
