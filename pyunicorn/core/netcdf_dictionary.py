#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2015 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)

"""
Provides classes for saving and loading NetCDF files from and to
appropriate Python dictionaries, allowing NetCDF4 compression methods.
"""

#
#  Imports
#

#  Import NumPy for the array object and fast numerics
import numpy as np

#  Import netCDF4 for Dataset class
try:
    from netCDF4 import Dataset
except:
    print "pyunicorn: Package netCDF4 could not be loaded. \
Some functionality in class NetCDFDictionary might not be available!"


#
#  Define class NetCDF
#

class NetCDFDictionary(object):

    """
    Encapsulates appropriate dictionary following NetCDF conventions.

    Also contains methods to load data from NetCDF and NetCDF4 files.

    :ivar file_name: (string) - The name of the data file.
    :ivar file_type: (string) - The format of the data file.
    :ivar observable_name: (string) - The short name of the observable within
                           data file (particularly relevant for NetCDF).
    :ivar observable_long_name: (string) - The long name of the observable
                                within data file.
    :ivar grid: (Grid) - The Grid object associated with the data.
    """

    # TODO: implement silence_level consistently
    def __init__(self, data_dict={
            "global_attributes": {
                "title": "Quasi-empty default dictionary"},
            "dimensions": {"x": 1},
            "variables": {
                "obs": {"array": np.array((1,)), "dims": ('x',),
                        "attributes": {"long_name": "observable"}}}},
            silence_level=0):
        """
        Return a NetCDF object containing an appropriately structured
        dictionary.

        If no data_dict is given, a default quasi-empty dictionary is created.

        :type data_dict: dictionary
        :arg data_dict: Contains data in a structure structure following
                        NetCDF conventions: {"global_attributes": {},
                        "dimensions": {}, "variables": {"obs": {"array": (),
                        "dims": (), "attributes": ()}}}

        :type silence_level: int >= 0
        :arg silence_level: The higher, the less progress info is output.
        """
        self.dict = data_dict

        self.silence_level = silence_level
        """(int >= 0)
        The higher, the less progress info is output.
        """

    def __str__(self):
        """Return a string representation of the object."""

        text = "Dictionary following NetCDF conventions."
        text += "\nGlobal attributes: " + str(self.dict["global_attributes"])
        text += "\nVariables:"
        for key in self.dict["variables"].keys():
            text += "\n\t" + key + "\t-> shape of array is:" \
                    + str(self.dict["variables"][key]["array"].shape)

        return text

    #
    #  Define methods for NetCDF4 files via NetCDF4 module
    #

    @staticmethod
    def from_file(file_name, with_array=['all']):
        """
        Load NetCDF4 file into a dictionary.

        Supported file types ``file_type`` are:
          - "NetCDF"
          - "NetCDF4"

        :arg str file_name: The name of the data file.
        :arg [str] with_array: Names of data arrays to be loaded completely.
        :rtype: NetCDF instance
        """
        #  Open NetCDF4 file
        try:
            cdf = Dataset(file_name, "r")
            print "MODULE: File %s opened." % file_name
        except:
            print "MODULE: File %s couldn't be opened." % file_name
            return

        #  Create empty dictionary structure
        content = {"global_attributes": {}, "dimensions": {}, "variables": {}}
        #  Copy all global attributes and all dimensions
        content["global_attributes"] = cdf.__dict__
        for dim_name, dim_obj in cdf.dimensions.iteritems():
            content["dimensions"][dim_name] = len(cdf.dimensions[dim_name])

        #  Loop over variables
        for var in cdf.variables.keys():
            #  Create empty dictionary for variable var
            content["variables"][var] = {"array": {}, "type": {}, "dims": {},
                                         "attributes": {}}
            #  Copy type, dimensions and variable attributes
            content["variables"][var]["type"] = cdf.variables[var].dtype.char
            content["variables"][var]["dims"] = cdf.variables[var].dimensions
            content["variables"][var]["attributes"] = \
                cdf.variables[var].__dict__

            #  Load data if wanted
            if var in with_array or 'all' in with_array:
                try:
                    content["variables"][var]["array"] = cdf.variables[var][:]
                    print "MODULE: Array %s loaded to dictionary." % var
                except MemoryError:
                    print "Memory Error during loading of array %s" % var
                except:
                    print "Other Error during loading of array %s" % var

                try:
                    content["variables"][var]["array"] = \
                        content["variables"][var]["array"].astype('float32')
                    print "MODULE: Array %s converted to 'float32'." % var
                except MemoryError:
                    print "MODULE: Memory Error during conversion of \
array %s." % var
                except:
                    print "MODULE: Other Error during conversion of \
array %s." % var

                #  If a scale_factor is given in the variable, rescale array
                if "scale_factor" in content["variables"][var]["attributes"]:
                    content["variables"][var]["array"] *= \
                        cdf.variables[var].scale_factor
                    del content["variables"][var]["attributes"]["scale_factor"]
                if "add_offset" in content["variables"][var]["attributes"]:
                    content["variables"][var]["array"] += \
                        cdf.variables[var].add_offset
                    del content["variables"][var]["attributes"]["add_offset"]
                #  Recalculate actual_range
                ar_max = content["variables"][var]["array"].max()
                ar_min = content["variables"][var]["array"].min()
                content["variables"][var]["attributes"]["actual_range"] = \
                    np.array([ar_min, ar_max])

        print "MODULE: Dictionary loaded from NetCDF file."
        cdf.close()

        return NetCDFDictionary(content)

    #  FIXME: createDimension - length of time variable should be "unlimited"
    #  TODO: Change file_name automatically if file already exists
    def to_file(self, file_name, compress=False, comp_level=6,
                least_significant_digit=10):
        """
        Write NetCDF4 file by using appropriate dictionary.

        :arg str file_name: The name of the data file.
        :arg bool compress: Determines whether the data should be compressed.
        :arg int comp_level: Level of compression, between 0 (no compression,
            fastest) and 9 (strongest compression, slowest).
        :arg int least_significant_digit: Last precise digit.
        """
        #  Check dictionary for empty entries
        for val in self.dict.keys():
            if len(self.dict[val]) == 0:
                print "MODULE: Entry %s is empty." % val

        print "MODULE: If %s already existed, old file will be \
overwritten." % file_name
        #  Format can be:
        #  NETCDF3_CLASSIC, NETCDF3_64BIT, NETCDF4_CLASSIC, NETCDF4
        cdf = Dataset(file_name, "w", format="NETCDF4")

        #  Write global attributes
        for val in self.dict["global_attributes"]:
            setattr(cdf, val, self.dict["global_attributes"][val])

        #  Write dimensions with given lengths
        for val in self.dict["dimensions"]:
            if val == "time":
                cdf.createDimension(val, self.dict["dimensions"][val])
            else:
                cdf.createDimension(val, self.dict["dimensions"][val])

        #  Write variables
        for var in self.dict["variables"]:
            #  Check variable dictionary for empty entries
            for key in self.dict["variables"][var].keys():
                if len(self.dict["variables"][var][key]) == 0 \
                        and key is not "type":
                    print "MODULE: Entry %s in variable %s is empty." \
                          % (key, var)

            var_type = self.dict["variables"][var]["array"].dtype.char
            try:
                var_ = cdf.createVariable(
                    var, var_type, self.dict["variables"][var]["dims"],
                    zlib=compress, complevel=comp_level,
                    least_significant_digit=least_significant_digit)
            except:
                print "MODULE: Couldn't create variable %s in NetCDF file." \
                      % var

            #  Copy the array
            var_[:] = self.dict["variables"][var]["array"]

            #  Calculate actual_range for variables
            ar_max = self.dict["variables"][var]["array"].max()
            ar_min = self.dict["variables"][var]["array"].min()
            self.dict["variables"][var]["attributes"]["actual_range"] = \
                np.array([ar_min, ar_max])

            #  Write all variable attributes to dictionary
            for att in self.dict["variables"][var]["attributes"]:
                setattr(var_, att,
                        self.dict["variables"][var]["attributes"][att])

        cdf.close()
        print "MODULE: Dictionary saved as NetCDF file %s." % file_name
