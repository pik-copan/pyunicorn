#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Tutorial on generating and analyzing lagged climate networks using Python.

Uses the Python packages ``core`` and ``climate`` providing all kinds of tools
related to climate networks. Written as part of a diploma / phd thesis in
Physics by Jonathan F. Donges (donges@pik-potsdam.de) at University of Potsdam
/ Humboldt University Berlin and Potsdam Institute of Climate Impact Research
(PIK),

Copyright 2008-2015.
"""

import numpy as np

from pyunicorn import CouplingAnalysis
from pyunicorn import climate

#
#  Settings
#

#  Related to data

#  Path and filename of NetCDF file containing climate data
DATA_FILENAME = "../../../Daten/Reanalysis/NCEP-NCAR/air.mon.mean.nc"

#  Type of data file ("NetCDF" indicates a NetCDF file with data on a regular
#  lat-lon grid, "iNetCDF" allows for arbitrary grids - > see documentation).
#  For example, the "NetCDF" FILE_TYPE is compatible with data from the IPCC
#  AR4 model ensemble or the reanalysis data provided by NCEP/NCAR.
FILE_TYPE = "NetCDF"

#  Indicate data source (optional)
DATA_SOURCE = "ncep_ncar_reanalysis"

#  Name of observable in NetCDF file ("air" indicates surface air temperature
#  in NCEP/NCAR reanalysis data)
OBSERVABLE_NAME = "air"

#  Select a subset in time and space from the data (e.g., a particular region
#  or a particular time window, or both)
WINDOW = {"time_min": 0., "time_max": 0., "lat_min": 0, "lon_min": 0,
          "lat_max": 30, "lon_max": 0}  # selects the whole data set

#  Indicate the length of the annual cycle in the data (e.g., 12 for monthly
#  data).  This is used for calculating climatological anomaly values
#  correctly.
TIME_CYCLE = 12

#  Related to climate network construction

#  For setting fixed threshold
THRESHOLD = 0.5

#  For setting fixed link density
LINK_DENSITY = 0.005

#  Set desired LAG
LAG = 10

#
#  Print script title
#

print "\n"
print "Tutorial on how to use climate"
print "-------------------------------"
print "\n"

#
#  Create a ClimateData object containing the data and print information
#

data = climate.ClimateData.Load(
    file_name=DATA_FILENAME, observable_name=OBSERVABLE_NAME,
    data_source=DATA_SOURCE, file_type=FILE_TYPE,
    window=WINDOW, time_cycle=TIME_CYCLE)

#  Print some information on the data set
print data

#
#  Create a MapPlots object to manage 2D-plotting on the sphere
#

#  Comment this if you are not using pyngl for plotting!!!
map_plots = climate.MapPlots(data.grid, DATA_SOURCE)

#
#  Generate lagged climate network using various procedures
#

#  Get array of anomaly time series
anomalies = data.get_anomaly()

#  Get data grid
grid = data.grid()

#  Create CouplingAnalysis object
ca = CouplingAnalysis(anomalies)

#  Compute cross correlation or mutual information matrix at specific lag
#  This may require large memory, since so far all 2*LAG + 1 lags are computed
#  and stored first from -LAG to +LAG
# similarity_matrix = \
#     np.abs(ca.get_cross_correlation(tau_max=LAG, lag_mode='all')[0, :, :])
similarity_matrix = \
    ca.get_mutual_information(bins=16, tau_max=LAG, lag_mode='all')[0, :, :]

#  This matrix is not symmetric in general. To create an undirected climate
#  network, symmetrization is necessary! This point is critical in interpreting
#  the results and some thought should be put into this.
similarity_matrix = 0.5 * (similarity_matrix + similarity_matrix.transpose())

#  Create the climate network
net = climate.ClimateNetwork(grid=grid, similarity_measure=similarity_matrix,
                             threshold=THRESHOLD)
# net = climate.ClimateNetwork(grid=grid, similarity_measure=similarity_matrix,
#                              link_density=LINK_DENSITY)

#
#  Some calculations
#

print "Link density:", net.link_density

#  Get degree
degree = net.degree()
#  Get closeness
closeness = net.closeness()
#  Get betweenness
betweenness = net.betweenness()
#  Get local clustering coefficient
clustering = net.local_clustering()
#  Get average link distance
ald = net.average_link_distance()
#  Get maximum link distance
mld = net.max_link_distance()

#
#  Save results to text file
#

#  Save the grid (mainly vertex coordinates) to text files
data.grid.save_txt(filename="grid.txt")

#  Save the degree sequence. Other measures may be saved similarly.
np.savetxt("degree.txt", degree)

#
#  Plotting
#

#  Comment everything below if you are not using pyngl for plotting!

#  Add network measures to the plotting queue
map_plots.add_dataset("Degree", degree)
map_plots.add_dataset("Closeness", closeness)
map_plots.add_dataset("Betweenness (log10)", np.log10(betweenness + 1))
map_plots.add_dataset("Clustering", clustering)
map_plots.add_dataset("Average link distance", ald)
map_plots.add_dataset("Maximum link distance", mld)

#  Change the map projection
map_plots.resources.mpProjection = "Robinson"
map_plots.resources.mpCenterLonF = 0

#  Change the levels of contouring
map_plots.resources.cnLevelSelectionMode = "EqualSpacedLevels"
map_plots.resources.cnMaxLevelCount = 20

# map_plots.resources.cnRasterSmoothingOn = True
# map_plots.resources.cnFillMode = "AreaFill"

map_plots.generate_map_plots(file_name="climate_network_measures",
                             title_on=False, labels_on=True)
