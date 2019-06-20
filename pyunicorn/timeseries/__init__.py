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
timeseries
==========

Provides classes for the analysis of dynamical systems and time series based on
recurrence plots, including measures of recurrence quantification analysis
(RQA) and recurrence network analysis.

Also contains methods to calculate the RQA (recurrence quantification analysis)
measures of the recurrence plot.

Related Publications
~~~~~~~~~~~~~~~~~~~~
Recurrence plots, RQA, Recurrence Network Analysis
--------------------------------------------------
[Marwan2007]_, [Marwan2009]_, [Donner2010a]_, [Donner2010b]_, [Donner2011a]_,
[Donner2011b]_, [Zou2010]_, [Donges2011b]_, [Donges2011c]_, [Donges2012]_,
[Feldhoff2012]_, [Zou2012]_, [Feldhoff2013]_

Visibility Graph Analysis of Geoscientific Time Series
------------------------------------------------------
[Donner2012]_

To do
~~~~~
  - Implement access to threshold_from_recurrence_rate_fast from
    constructor of RecurrencePlot.
  - Add more RQA measures and improve their documentation, including formulas.
  - Add random node relabeling to Network class.

Known Bugs
~~~~~~~~~~
  - RQA measures are not yet calculated correctly for Eckmann recurrence plots.

"""

from .cross_recurrence_plot import CrossRecurrencePlot
from .inter_system_recurrence_network import InterSystemRecurrenceNetwork
from .joint_recurrence_network import JointRecurrenceNetwork
from .joint_recurrence_plot import JointRecurrencePlot
from .recurrence_network import RecurrenceNetwork
from .recurrence_plot import RecurrencePlot
from .surrogates import Surrogates
from .visibility_graph import VisibilityGraph
