# This file is part of pyunicorn.
# Copyright (C) 2008--2024 Jonathan F. Donges and pyunicorn authors
# URL: <https://www.pik-potsdam.de/members/donges/software-2/software>
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
eventseries
===========

Provides a class for analyzing event series, namely event synchronization and
event coincidence analysis

Related Publications
~~~~~~~~~~~~~~~~~~~~
[Quiroga2002]_, [Boers2014]_, [Donges2016]_, [Odenweller2020]_, [Kreuz2007]_,
[Marwan2015]_, [Schleussner2016]_.

To do
~~~~~
  - Combine precursor and trigger coincidence rate to obtain one ECA measure
"""

from .event_series import EventSeries
