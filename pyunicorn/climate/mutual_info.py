#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2015 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)

"""
Provides classes for generating and analyzing complex climate networks.
"""

#
#  Import essential packages
#

# array object and fast numerics
import numpy as np
# C++ inline code
import weave

#  Import progress bar for easy progress bar handling
from ..utils import progressbar

#  Import cnNetwork for Network base class
from .climate_network import ClimateNetwork


#
#  Define class MutualInfoClimateNetwork
#

class MutualInfoClimateNetwork(ClimateNetwork):

    """
    Represents a mutual information climate network.

    Constructs a static climate network based on mutual information at zero
    lag, as in [Ueoka2008]_.

    Mutual information climate networks are undirected, since mutual
    information is a symmetrical measure. In contrast to Pearson correlation
    used in :class:`.TsonisClimateNetwork`, mutual information has the
    potential to detect nonlinear statistical interdependencies.

    Derived from the class ClimateNetwork.
    """

    #
    #  Defines internal methods
    #

    def __init__(self, data, threshold=None, link_density=None,
                 non_local=False, node_weight_type="surface", winter_only=True,
                 silence_level=0):
        """
        Initialize an instance of MutualInfoClimateNework.

        .. note::
           Either threshold **OR** link_density have to be given!

        Possible choices for ``node_weight_type``:
          - None (constant unit weights)
          - "surface" (cos lat)
          - "irrigation" (cos**2 lat)

        :type data: :class:`.ClimateData`
        :arg data: The climate data used for network construction.

        :arg float threshold: The threshold of similarity measure, above which
            two nodes are linked in the network.
        :arg float link_density: The networks's desired link density.
        :arg bool non_local: Determines, whether links between spatially close
            nodes should be suppressed.
        :arg str node_weight_type: The type of geographical node weight to be
            used.
        :arg bool winter_only: Determines, whether only data points from the
            winter months (December, January and February) should be used for
            analysis. Possibly, this further suppresses the annual cycle in the
            time series.
        :arg int silence_level: The inverse level of verbosity of the object.
        """
        if silence_level <= 1:
            print "Generating a mutual information climate network..."

        #  Set instance variables
        self.data = data
        """(ClimateData) - The climate data used for network construction."""

        self.N = self.data.grid.N
        self._threshold = threshold
        self._prescribed_link_density = link_density
        self._non_local = non_local
        self.node_weight_type = node_weight_type
        self.silence_level = silence_level

        #  Class specific settings
        self.mi_file = "mutual_information_" + data.data_source + "_" \
            + data.observable_name + ".data"
        """(string) - The name of the file for storing the mutual information
                      matrix."""

        #  The constructor of ClimateNetwork is called within this method.
        self.set_winter_only(winter_only)

    def __str__(self):
        """
        Return a string representation of MutualInformationClimateNetwork.
        """
        text = "Mutual information climate network: \n"
        text += ClimateNetwork.__str__(self)

        return text

    #
    #  Defines methods to calculate the mutual information matrix
    #

    def eval_weave_calculate_mutual_information(self, anomaly):
        """
        Compare the fast and slow weave code to calculate mutual information.

        :type anomaly: 2D Numpy array (time, index)
        :arg anomaly: The anomaly time series.

        :rtype: tuple of two 2D Numpy arrays (index, index)
        :return: the mutual information matrices from fast and slow algorithm.
        """
        mi_fast = self._weave_calculate_mutual_information(anomaly, fast=True)
        mi_slow = self._weave_calculate_mutual_information(anomaly, fast=False)

        return (mi_fast, mi_slow)

    def _weave_calculate_mutual_information(self, anomaly, n_bins=32,
                                            fast=True):
        """
        Calculate the mutual information matrix at zero lag.

        The weave code is adopted from the Tisean 3.0.1 mutual.c module.

        :type anomaly: 2D Numpy array (time, index)
        :arg anomaly: The anomaly time series.

        :arg int n_bins: The number of bins for estimating probability
            distributions.
        :arg bool fast: Indicates, whether fast or slow algorithm should be
            used.
        :rtype: 2D array (index, index)
        :return: the mutual information matrix at zero lag.
        """
        if self.silence_level <= 1:
            print "Calculating mutual information matrix at zero lag from \
anomaly values using Weave..."

        #  Normalize anomaly time series to zero mean and unit variance
        self.data.normalize_time_series_array(anomaly)

        #  Create local transposed copy of anomaly
        anomaly = np.fastCopyAndTranspose(anomaly)

        (N, n_samples) = anomaly.shape

        #  Get common range for all histograms
        range_min = float(anomaly.min())
        range_max = float(anomaly.max())

        #  Rescale all time series to the interval [0,1],
        #  using the maximum range of the whole dataset.
        scaling = float(1. / (range_max - range_min))

        #  Create array to hold symbolic trajectories
        symbolic = np.empty(anomaly.shape, dtype="int32")

        #  Initialize array to hold 1d-histograms of individual time series
        hist = np.zeros((N, n_bins), dtype="int32")

        #  Initialize array to hold 2d-histogram for one pair of time series
        hist2d = np.zeros((n_bins, n_bins), dtype="int32")

        #  Initialize mutual information array
        mi = np.zeros((N, N), dtype="float32")

        code = r"""
        int i, j, k, l, m;
        int symbol, symbol_i, symbol_j;
        double norm, rescaled, hpl, hpm, plm;

        //  Calculate histogram norm
        norm = 1.0 / n_samples;

        for (i = 0; i < N; i++) {
            for (k = 0; k < n_samples; k++) {

                //  Calculate symbolic trajectories for each time series,
                //  where the symbols are bins.
                rescaled = scaling * (anomaly(i,k) - range_min);

                if (rescaled < 1.0) {
                    symbolic(i,k) = rescaled * n_bins;
                }
                else {
                    symbolic(i,k) = n_bins - 1;
                }

                //  Calculate 1d-histograms for single time series
                symbol = symbolic(i,k);
                hist(i,symbol) += 1;
            }
        }

        for (i = 0; i < N; i++) {
            for (j = 0; j <= i; j++) {

                //  The case i = j is not of interest here!
                if (i != j) {
                    //  Calculate 2d-histogram for one pair of time series
                    //  (i,j).
                    for (k = 0; k < n_samples; k++) {
                        symbol_i = symbolic(i,k);
                        symbol_j = symbolic(j,k);
                        hist2d(symbol_i,symbol_j) += 1;
                    }

                    //  Calculate mutual information for one pair of time
                    //  series (i,j).
                    for (l = 0; l < n_bins; l++) {
                        hpl = hist(i,l) * norm;
                        if (hpl > 0.0) {
                            for (m = 0; m < n_bins; m++) {
                                hpm = hist(j,m) * norm;
                                if (hpm > 0.0) {
                                    plm = hist2d(l,m) * norm;
                                    if (plm > 0.0) {
                                        mi(i,j) += plm * log(plm/hpm/hpl);
                                    }
                                }
                            }
                        }
                    }

                    //  Symmetrize MI
                    mi(j,i) = mi(i,j);

                    //  Reset hist2d to zero in all bins
                    for (l = 0; l < n_bins; l++) {
                        for (m = 0; m < n_bins; m++) {
                            hist2d(l,m) = 0;
                        }
                    }
                }
            }
        }
        """

        # anomaly must be a contiguous Numpy array for this code to work
        # correctly! All the other arrays are generated from scratch in this
        # method and are guaranteed to be contiguous by Numpy.
        fastCode = r"""
        long i, j, k, l, m, in_bins, jn_bins, ln_bins, in_samples, jn_samples,
             in_nodes;
        double norm, rescaled, hpl, hpm, plm;

        double *p_anomaly;
        float *p_mi, *p_mi2;
        long *p_symbolic, *p_symbolic1, *p_symbolic2, *p_hist, *p_hist1,
             *p_hist2, *p_hist2d;

        //  Calculate histogram norm
        norm = 1.0 / n_samples;

        //  Initialize in_samples, in_bins
        in_samples = in_bins = 0;

        for (i = 0; i < N; i++) {

            //  Set pointer to anomaly(i,0)
            p_anomaly = anomaly + in_samples;
            //  Set pointer to symbolic(i,0)
            p_symbolic = symbolic + in_samples;

            for (k = 0; k < n_samples; k++) {

                //  Rescale sample into interval [0,1]
                rescaled = scaling * (*p_anomaly - range_min);

                //  Calculate symbolic trajectories for each time series,
                //  where the symbols are bin numbers.
                if (rescaled < 1.0) {
                    *p_symbolic = rescaled * n_bins;
                }
                else {
                    *p_symbolic = n_bins - 1;
                }

                //  Calculate 1d-histograms for single time series
                //  Set pointer to hist(i, *p_symbolic)
                p_hist = hist + in_bins + *p_symbolic;
                (*p_hist)++;

                //  Set pointer to anomaly(k+1,i)
                p_anomaly++;
                //  Set pointer to symbolic(k+1,i)
                p_symbolic++;
            }
            in_samples += n_samples;
            in_bins += n_bins;
        }

        //  Initialize in_samples, in_bins, in_nodes
        in_samples = in_bins = in_nodes = 0;

        for (i = 0; i < N; i++) {

            //  Set pointer to mi(i,0)
            p_mi = mi + in_nodes;
            //  Set pointer to mi(0,i)
            p_mi2 = mi + i;

            //  Initialize jn_samples, jn_bins
            jn_samples = jn_bins = 0;

            for (j = 0; j <= i; j++) {

                //  Don't do anything for i = j, this case is not of
                //  interest here!
                if (i != j) {

                    //  Set pointer to symbolic(i,0)
                    p_symbolic1 = symbolic + in_samples;
                    //  Set pointer to symbolic(j,0)
                    p_symbolic2 = symbolic + jn_samples;

                    //  Calculate 2d-histogram for one pair of time series
                    //  (i,j).
                    for (k = 0; k < n_samples; k++) {

                        //  Set pointer to hist2d(*p_symbolic1, *p_symbolic2)
                        p_hist2d = hist2d + (*p_symbolic1)*n_bins
                                   + *p_symbolic2;

                        (*p_hist2d)++;

                        //  Set pointer to symbolic(i,k+1)
                        p_symbolic1++;
                        //  Set pointer to symbolic(j,k+1)
                        p_symbolic2++;
                    }

                    //  Calculate mutual information for one pair of time
                    //  series (i,j).

                    //  Set pointer to hist(i,0)
                    p_hist1 = hist + in_bins;

                    //  Initialize ln_bins
                    ln_bins = 0;

                    for (l = 0; l < n_bins; l++) {

                        //  Set pointer to hist(j,0)
                        p_hist2 = hist + jn_bins;
                        //  Set pointer to hist2d(l,0)
                        p_hist2d = hist2d + ln_bins;

                        hpl = (*p_hist1) * norm;

                        if (hpl > 0.0) {
                            for (m = 0; m < n_bins; m++) {

                                hpm = (*p_hist2) * norm;

                                if (hpm > 0.0) {
                                    plm = (*p_hist2d) * norm;
                                    if (plm > 0.0) {
                                        *p_mi += plm * log(plm/hpm/hpl);
                                    }
                                }

                                //  Set pointer to hist(j,m+1)
                                p_hist2++;
                                //  Set pointer to hist2d(l,m+1)
                                p_hist2d++;
                            }
                        }
                        //  Set pointer to hist(i,l+1)
                        p_hist1++;

                        ln_bins += n_bins;
                    }

                    //  Symmetrize MI
                    *p_mi2 = *p_mi;

                    //  Initialize ln_bins
                    ln_bins = 0;

                    //  Reset hist2d to zero in all bins
                    for (l = 0; l < n_bins; l++) {

                        //  Set pointer to hist2d(l,0)
                        p_hist2d = hist2d + ln_bins;

                        for (m = 0; m < n_bins; m++) {
                            *p_hist2d = 0;

                            //  Set pointer to hist2d(l,m+1)
                            p_hist2d++;
                        }
                        ln_bins += n_bins;
                    }
                }
                //  Set pointer to mi(i,j+1)
                p_mi++;
                //  Set pointer to mi(j+1,i)
                p_mi2 += N;

                jn_samples += n_samples;
                jn_bins += n_bins;
            }
            in_samples += n_samples;
            in_bins += n_bins;
            in_nodes += N;
        }
        """
        args = ['anomaly', 'n_samples', 'N', 'n_bins', 'scaling', 'range_min',
                'symbolic', 'hist', 'hist2d', 'mi']

        if fast:
            weave.inline(fastCode, arg_names=args, compiler='gcc',
                         extra_compile_args=['-O3'])
        else:
            weave.inline(code, arg_names=args,
                         type_converters=weave.converters.blitz,
                         compiler='gcc', extra_compile_args=['-O3'])

        if self.silence_level <= 1:
            print "Done!"

        return mi

    def _calculate_mutual_information(self, anomaly, n_bins=32):
        """
        Calculate the mutual information matrix at zero lag.

        .. note::
           Slow since solely based on Python and Numpy!

        :type anomaly: 2D array (time, index)
        :arg anomaly: The anomaly time series.
        :arg int n_bins: The number of bins for estimating probability
                     distributions.
        :rtype: 2D array (index, index)
        :return: the mutual information matrix at zero lag.
        """
        if self.silence_level <= 1:
            print "Calculating mutual information matrix at zero lag from \
anomaly values..."

        #  Define references to numpy functions for faster function calls
        histogram = np.histogram
        histogram2d = np.histogram2d
        log = np.log

        #  Normalize anomaly time series to zero mean and unit variance
        self.data.normalize_time_series_array(anomaly)

        #  Get faster reference to length of time series = number of samples
        #  per grid point.
        n_samples = anomaly.shape[0]

        #  Initialize mutual information array
        mi = np.zeros((self.N, self.N))

        #  Get common range for all histograms
        range_min = anomaly.min()
        range_max = anomaly.max()

        #  Calculate the histograms for each time series
        p = np.zeros((self.N, n_bins))

        for i in xrange(self.N):
            p[i, :] = (histogram(anomaly[:, i], bins=n_bins,
                       range=(range_min, range_max))[0]).astype("float64")

        #  Normalize by total number of samples = length of each time series
        p /= n_samples

        #  Make sure that bins with zero estimated probability are not counted
        #  in the entropy measures.
        p[p == 0] = 1

        #  Compute the information entropies of each time series
        H = - (p * log(p)).sum(axis=1)

        #  Initialize progress bar
        if self.silence_level <= 1:
            progress = progressbar.ProgressBar(maxval=self.N**2).start()

        #  Calculate only the lower half of the MI matrix, since MI is
        #  symmetric with respect to X and Y.
        for i in xrange(self.N):
            #  Update progress bar every 10 steps
            if self.silence_level <= 1:
                if (i % 10) == 0:
                    progress.update(i**2)

            for j in xrange(i):
                #  Calculate the joint probability distribution
                pxy = (histogram2d(anomaly[:, i], anomaly[:, j], bins=n_bins,
                       range=((range_min, range_max),
                              (range_min, range_max)))[0]).astype("float64")

                #  Normalize joint distribution
                pxy /= n_samples

                #  Compute the joint information entropy
                pxy[pxy == 0] = 1
                HXY = - (pxy * log(pxy)).sum()

                #  ... and store the result
                mi.itemset((i, j), H.item(i) + H.item(j) - HXY)
                mi.itemset((j, i), mi.item((i, j)))

        if self.silence_level <= 1:
            progress.finish()

        return mi

    def calculate_similarity_measure(self, anomaly):
        """
        Calculate the mutual information matrix.

        Encapsulates calculation of mutual information with standard
        parameters.

        :type anomaly: 2D Numpy array (time, index)
        :arg anomaly: The anomaly time series.

        :rtype: 2D Numpy array (index, index)
        :return: the mutual information matrix at zero lag.
        """
        return self._weave_calculate_mutual_information(anomaly)

    def mutual_information(self, anomaly):
        """
        Return mutual information matrix at zero lag.

        Check if mutual information matrix (MI) was already calculated before:
          - If yes, return MI from a data file.
          - If not, return MI from calculation and store in file.

        :type anomaly: 2D Numpy array (time, index)
        :arg anomaly: The anomaly time series.

        :rtype: 2D Numpy array (index, index)
        :return: the mutual information matrix at zero lag.
        """

        try:
            #  Try to load MI from file
            if self.silence_level <= 1:
                print "Loading mutual information matrix from", \
                      self.mi_file, "..."

            f = file(self.mi_file)
            mi = np.load(f)

            #  Check if the dimensions of mutual_information correspond to the
            #  grid.
            if mi.shape != (self.N, self.N):
                print self.mi_file, "in current directory has incorrect \
dimensions!"
                raise

        except:
            if self.silence_level <= 1:
                print "An error occured while loading data from", \
                      self.mi_file, "."
                print "Recalculating mutual information and storing in", \
                      self.mi_file

            f = file(self.mi_file, "w")
            mi = self._weave_calculate_mutual_information(anomaly)
            mi.dump(f)
            f.close()

        return mi

    def winter_only(self):
        """
        Indicate, if only winter months were used for network generation.

        :return bool: whether only winter months were used for network
            generation.
        """
        return self._winter_only

    def set_winter_only(self, winter_only):
        """
        Toggle use of exclusively winter data points for network generation.

        Also explicitly re(generates) the instance of MutualInfoClimateNetwork.

        :arg bool winter_only: Indicates, whether only winter months were used
            for network generation.
        """
        self._winter_only = winter_only

        if winter_only:
            winter_anomaly = self.data.anomaly_selected_months([0, 1, 11])
            mi = self.mutual_information(winter_anomaly)
        else:
            mi = self.mutual_information(self.data.anomaly())
            # correlationMeasure = self.\
            #     _weaveCalculateMIMatrix(self.data.getCycleAnomaly())

        #  Call the constructor of the parent class ClimateNetwork
        ClimateNetwork.__init__(self, grid=self.data.grid,
                                similarity_measure=mi,
                                threshold=self.threshold(),
                                link_density=self._prescribed_link_density,
                                non_local=self.non_local(),
                                directed=False,
                                node_weight_type=self.node_weight_type,
                                silence_level=self.silence_level)

    #
    #  Defines methods to calculate  weighted network measures
    #

    def mutual_information_weighted_average_path_length(self):
        """
        Return mutual information weighted average path length.

        :return float: the mutual information weighted average path length.
        """
        if "mutual_information" not in self._path_lengths_cached:
            self.set_link_attribute("mutual_information",
                                    abs(self.mutual_information()))

        return self.average_path_length("mutual_information")

    def mutual_information_weighted_closeness(self):
        """
        Return mutual information weighted closeness.

        :rtype: 1D Numpy array [index]
        :return: the mutual information weighted closeness sequence.
        """
        if "mutual_information" not in self._path_lengths_cached:
            self.set_link_attribute("mutual_information",
                                    abs(self.mutual_information()))

        return self.closeness("mutual_information")

    def local_mutual_information_weighted_vulnerability(self):
        """
        Return mutual information weighted vulnerability.

        :rtype: 1D Numpy array [index]
        :return: the mutual information weighted vulnerability sequence.
        """
        if "mutual_information" not in self._path_lengths_cached:
            self.set_link_attribute("mutual_information",
                                    abs(self.mutual_information()))

        return self.local_vulnerability("mutual_information")
