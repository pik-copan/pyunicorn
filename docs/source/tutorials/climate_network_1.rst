
Constructing and analyzing a climate network
============================================

This tutorials illustrates the use of ``climate`` for constructing a
climate network from given data in a commonly used format, performing a
statistical analysis of the network and finally plotting the results on a map.

For example, our software can handle data from the NCEP/NCAR reanalysis 1
project like this monthly surface air temperature data set (a NetCDF file):
ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.derived/surface/air.mon.mean.nc

You can use ``PyNgl`` for plotting the results on maps
(http://www.pyngl.ucar.edu/Download/). Alternatively, the tutorial saves the
results as well as the grid information in text files which can be used for
plotting in your favorite software.

This tutorial is also available as an ipython notebook.

.. literalinclude:: ../../../examples/tutorials/climate_network.py
