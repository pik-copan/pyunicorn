#!/usr/bin/python

"""
Tutorial on analyzing climate networks using Python.

Uses the Python packages ``core`` and ``climate`` providing all kinds of tools
related to climate networks. Written as part of a diploma / phd thesis in
Physics by Jonathan F. Donges (donges@pik-potsdam.de) at University of Potsdam
/ Humboldt University Berlin and Potsdam Institute of Climate Impact Research
(PIK),

Copyright 2008-2023.
"""

import numpy as np
from matplotlib import pyplot as plt

from pyunicorn import climate

import cartopy.crs as ccrs
import cartopy.feature as cf

#
#  Settings
#

#  Related to data

#  Download the data set from the link that is printed below and copy it to the
#  directory of this script or change the path to the location of the data set
LINK = "\nData available at: https://www.esrl.noaa.gov/psd/repository/" + \
       "entry/show?entryid=0def76a0-9b32-47a4-8bc3-c4977c67ed95"
print(LINK)

DATA_FILENAME = "./air.mon.mean.nc"

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
#  Plase note that "time_min" and "time_max" have to be in the time format of
#  NetCDF files, e.g. 1918248. to 1921128. for the NetCDF file used here.
#  In particular, they are not indices, e.g. "time_max"=10. will not work.
WINDOW = {"time_min": 0., "time_max": 0., "lat_min": 0, "lon_min": 0,
          "lat_max": 30, "lon_max": 0}  # selects the whole data set

#  Indicate the length of the annual cycle in the data (e.g., 12 for monthly
#  data). This is used for calculating climatological anomaly values
#  correctly.
TIME_CYCLE = 12

#  Related to climate network construction

#  For setting fixed threshold
THRESHOLD = 0.5

#  For setting fixed link density
LINK_DENSITY = 0.005

#  Indicates whether to use only data from winter months (DJF) for calculating
#  correlations
WINTER_ONLY = False

#
#  Print script title
#

print("\n")
print("Tutorial on how to use climate")
print("-------------------------------")
print("\n")

#
#  Create a ClimateData object containing the data and print information
#

data = climate.ClimateData.Load(
    file_name=DATA_FILENAME, observable_name=OBSERVABLE_NAME,
    data_source=DATA_SOURCE, file_type=FILE_TYPE,
    window=WINDOW, time_cycle=TIME_CYCLE)

#  Print some information on the data set

data.grid.print_grid_size()


#
#  Generate climate network using various procedures
#

#  One of several alternative similarity measures and construction mechanisms
#  may be chosen here

#  Create a climate network based on Pearson correlation without lag and with
#  fixed threshold
net = climate.TsonisClimateNetwork(
    data, threshold=THRESHOLD, winter_only=WINTER_ONLY)

print("net created")

#  Create a climate network based on Pearson correlation without lag and with
#  fixed link density
# net = climate.TsonisClimateNetwork(
#     data, link_density=LINK_DENSITY, winter_only=WINTER_ONLY)

#  Create a climate network based on Spearman's rank order correlation without
#  lag and with fixed threshold
# net = climate.SpearmanClimateNetwork(
#     data, threshold=THRESHOLD, winter_only=WINTER_ONLY)

#  Create a climate network based on mutual information without lag and with
#  fixed threshold
# net = climate.MutualInfoClimateNetwork(
#     data, threshold=THRESHOLD, winter_only=WINTER_ONLY)

#
#  Some calculations
#

print("Link density:", net.link_density)

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

# create a Cartopy plot instance called cn_plot (cn for climate network)
# from the data with tite DATA_SOURCE
cn_plot = climate.CartopyPlots(data.grid, DATA_SOURCE)

#  Add network measures to the plotting queue
cn_plot.add_dataset("Degree", degree)
cn_plot.add_dataset("Closeness", closeness)
cn_plot.add_dataset("Betweenness (log10)", np.log10(betweenness + 1))
cn_plot.add_dataset("Clustering", clustering)
cn_plot.add_dataset("Average link distance", ald)
cn_plot.add_dataset("Maximum link distance", mld)

# before plotting, we can change some things, like cmap
ax = plt.set_cmap('plasma')

# Plot with cartopy and matplotlib
cn_plot.generate_plots(file_name="climate_network_measures",
                                 title_on=False, labels_on=True)
