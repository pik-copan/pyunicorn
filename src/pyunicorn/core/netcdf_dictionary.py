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
Provides classes for saving and loading NetCDF files from and to
appropriate Python dictionaries, allowing NetCDF4 compression methods.
"""

#
#  Imports
#

import numpy as np

try:
    from h5netcdf.legacyapi import Dataset
except ImportError:
    try:
        from netCDF4 import Dataset
    except ImportError:
        print("pyunicorn: Packages netCDF4 or h5netcdf could not be loaded. "
              "Some functionality in class NetCDFDictionary might not be "
              "available!")


#
#  Define class NetCDF
#

class NetCDFDictionary:

    """
    Encapsulates appropriate dictionary following NetCDF conventions.

    Also contains methods to load data from NetCDF and NetCDF4 files.
    """

    # TODO: implement silence_level consistently
    def __init__(self, data_dict=None, silence_level=0):
        """
        Return a NetCDF object containing an appropriately structured
        dictionary.

        If no data_dict is given, a default quasi-empty dictionary is created.

        :type data_dict: dictionary
        :arg data_dict: Contains data in a structure following NetCDF
            conventions: {"global_attributes": {}, "dimensions": {},
            "variables": {"obs": {"array": (), "dims": (), "attributes": ()}}}

        :type silence_level: int >= 0
        :arg silence_level: The higher, the less progress info is output.
        """
        if data_dict is None:
            data_dict = {
                "global_attributes": {
                    "title": "Quasi-empty default dictionary"},
                "dimensions": {"x": 1},
                "variables": {
                    "obs": {"array": np.array((1,)), "dims": ('x',),
                            "attributes": {"long_name": "observable"}}}}
        self.dict = data_dict

        self.silence_level = silence_level
        """(int >= 0)
        The higher, the less progress info is output.
        """

    def __str__(self):
        """
        Return a string representation of the object.
        """
        text = (f'NetCDFDictionary:\nGlobal attributes:\n'
                f'{self.dict["global_attributes"]}\nVariables:')

        for key in self.dict["variables"].keys():
            text += (f'\n\t{key}\t-> array shape'
                     f'{self.dict["variables"][key]["array"].shape}')
        return text

    #
    #  Define methods for NetCDF4 files via NetCDF4 module
    #

    @staticmethod
    def from_file(file_name, with_array='all'):
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
            print(f"MODULE: File {file_name} opened.")
        except RuntimeError:
            print(f"MODULE: File {file_name} couldn't be opened.")
            return None

        #  Create empty dictionary structure
        content = {"global_attributes": {}, "dimensions": {}, "variables": {}}
        #  Copy all global attributes and all dimensions
        content["global_attributes"] = cdf.__dict__
        for dim_name, _ in cdf.dimensions.iteritems():
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
            if var in with_array or with_array == 'all':
                try:
                    content["variables"][var]["array"] = cdf.variables[var][:]
                    print(f"MODULE: Array {var} loaded to dictionary.")
                except MemoryError:
                    print(f"Memory Error during loading of array {var}")
                except RuntimeError:
                    print(f"Other Error during loading of array {var}")

                try:
                    content["variables"][var]["array"] = \
                        content["variables"][var]["array"].astype('float32')
                    print(f"MODULE: Array {var} converted to 'float32'.")
                except MemoryError:
                    print("MODULE: Memory Error during conversion of "
                          f"array {var}.")
                except RuntimeError:
                    print("MODULE: Other Error during conversion of "
                          f"array {var}.")

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

        print("MODULE: Dictionary loaded from NetCDF file.")
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
            if not self.dict[val]:
                print("MODULE: Entry {val} is empty.")

        print(f"MODULE: If {file_name} already existed, old file will be "
              "overwritten.")
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
                if not self.dict["variables"][var][key] and key != "type":
                    print(f"MODULE: Entry {key} in variable {var} is empty.")

            var_type = self.dict["variables"][var]["array"].dtype.char
            try:
                var_ = cdf.createVariable(
                    var, var_type, self.dict["variables"][var]["dims"],
                    zlib=compress, complevel=comp_level,
                    least_significant_digit=least_significant_digit)
            except RuntimeError:
                print(f"MODULE: Couldn't create variable {var} "
                      "in NetCDF file.")

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
        print("MODULE: Dictionary saved as NetCDF file {file_name}.")
