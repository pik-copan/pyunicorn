
Package Overview
================

A brief introduction to the methods, measures and algorithms provided by
``pyunicorn``.

General complex networks
------------------------
Many standard complex network measures, network models and algorithms are
supported, most of them inherited from the ``igraph`` package, e.g., degree,
closeness and betweenness centralities, clustering coefficient and
transitivity or commmunity detection algorithms and network models such as
Erdos-Renyi or Barabasi-Albert. Moreover, a number of less common network
statistics like Newman's or Arenas' random walk betweenness can be computed.
Reading and saving network data from and to many common data formats is
possible.

* :doc:`api/core/network`

Spatially embedded networks
---------------------------
``pyunicorn`` includes measures and models specifically designed for spatially
embedded networks (or simply spatial networks) via the ``GeoNetwork`` and ``Grid``
classes.

* :doc:`api/core/geo_network`
* :doc:`api/core/grid`

Interacting/multiplex networks (networks of networks)
-----------------------------------------------------
The ``InteractingNetworks`` class provides a rich collection of network measures and models specifically designed for investigating the structure of networks of
networks (also called interacting networks, interdependent networks or
multiplex networks in different contexts). Examples include the cross-link
density of connections between different subnetworks or the cross-shortest
path betweenness quantifying the importance of nodes for mediating interactions
between different subnetworks. Models of interacting networks allow to assess
the degree of organization of the cross-connectivity between subnetworks.

* :doc:`api/core/interacting_networks`

Node-weighted (node-splitting invariant) network measures
---------------------------------------------------------
Node-weighted networks measures derived following the node-splitting invariance
approach are useful for studying systems with nodes representing subsystems of
heterogeneous size, weight, area, volume or importance, e.g., nodes
representing grid cells of widely different area in climate networks or voxels
of differing volume in functional brain networks. ``pyunicorn`` provides
node-weighted variants of most standard and non-standard measures for networks
as well as interacting networks.

* :doc:`api/core/network`
* :doc:`api/core/interacting_networks`

(Coupled) Climate networks
--------------------------
``pyunicorn`` provides classes for the easy construction and analysis of the
statistical interdependency structure within and between fields of time series (functional networks) using various similarity measures such as Pearson and Spearman correlation, lagged linear correlation, mutual information and event
synchronization. Climate networks allow the analysis of single fields of time series, whereas coupled climate networks focus on studying the
interrelationships between two fields of time series. While there is a
historical focus on applications to climate data, those methods can also be
applied to other sources of time series data such as neuroscientific (e.g.,
FMRI and EEG data) or financial data (e.g., stock market indices).

* :doc:`api/climate/climate_network`
* :doc:`api/climate/coupled_climate_network`
* :doc:`api/climate/climate_data`

Recurrence quantification/network analysis
------------------------------------------
Recurrence analysis is a powerful method for studying nonlinear systems,
particularly based on univariate and multivariate time series data. Recurrence
quantification analysis (RQA) and recurrence network analysis (RNA) allow to
classify different dynamical regimes in time series and to detect regime
shifts, dynamical transitions or tipping points, among many other applications.
Bivariate methods such as joint recurrence plots/networks, cross recurrence
plots or inter system recurrence networks allow to investigate the coupling
structure between two dynamical systems based on time series, including methods
to detect the directionality of coupling. Recurrence analysis is applicable to
general time series data from many fields such as climatology,
paleoclimatology, medicine, neuroscience or economics.

* :doc:`api/timeseries/recurrence_plot`
* :doc:`api/timeseries/recurrence_network`
* :doc:`api/timeseries/joint_recurrence_plot`
* :doc:`api/timeseries/joint_recurrence_network`
* :doc:`api/timeseries/cross_recurrence_plot`
* :doc:`api/timeseries/inter_system_recurrence_network`

Visibility graph analysis
-------------------------
Visibility graph analysis is an alternative approach to nonlinear time series
analysis, allowing to study among others fractal properties and long-term memory in time series. As a special feature, ``pyunicorn`` provides
time-directed measures such as advanced and retarded degree/clustering that can
be used for designing tests for time-irreversibility (time-reversal
asymmetry) of processes.

* :doc:`api/timeseries/visibility_graph`

Surrogate time series
---------------------
Surrogate time series are useful for testing hypothesis on observed time series
properties, e.g., on what features of a time series are expected to arise with
high probability for randomized time series with the same autocorrelation
structure. ``pyunicorn`` can be used to generate various types of time series
surrogates, including white noise surrogates, Fourier surrogates, amplitude adjusted Fourier (AAFT) surrogates or twin surrogates (conserving the recurrence structure of the underlying time series).

* :doc:`api/timeseries/surrogates`
