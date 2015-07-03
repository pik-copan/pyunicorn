#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2015 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)

"""
Provides classes for the analysis of dynamical systems and time series based
on recurrence plots, including measures of recurrence quantification
analysis (RQA) and recurrence network analysis.
"""

# array object and fast numerics
import numpy as np

from .. import InteractingNetworks, weave_inline


#
#  Class definitions
#

class VisibilityGraph(InteractingNetworks):
    """
    Class VisibilityGraph for generating and analyzing visibility graphs of
    time series.

    Visibility graphs were initially applied for time series analysis by
    [Lacasa2008]_.
    """

    #
    #  Internal methods
    #

    def __init__(self, time_series, timings=None, missing_values=False,
                 horizontal=False, silence_level=0):
        """
        Missing values are handled as infinite values, effectively separating
        the visibility graph into different disconnected components.

        .. note::
           Missing values have to be marked by the Numpy NaN flag!

        :type time_series: 2D array (time, dimension)
        :arg time_series: The time series to be analyzed, can be scalar or
            multi-dimensional.
        :arg str timings: Timings of the observations in :attr:`time_series`.
        :arg bool missing_values: Toggle special treatment of missing values in
            :attr:`time_series`.
        :arg bool horizontal: Indicates whether a horizontal visibility
            relation is used.
        :arg number silence_level: Inverse level of verbosity of the object.
        """
        #  Set silence_level
        self.silence_level = silence_level
        """The inverse level of verbosity of the object."""

        #  Set missing_values flag
        self.missing_values = missing_values
        """Controls special treatment of missing values in
           :attr:`time_series`."""

        #  Store time series
        self.time_series = time_series.copy().astype("float32")
        """The time series from which the visibility graph is constructed."""

        if timings is not None:
            timings = timings.copy().astype("float32")
        else:
            timings = np.arange(len(time_series), dtype="float32")

        #  Store timings
        self.timings = timings
        """The timimgs of the time series data points."""

        #  Get missing value indices
        if self.missing_values:
            self.missing_value_indices = np.isnan(self.time_series)

        #  Determine visibility relations
        if not horizontal:
            A = self.visibility_relations()
        else:
            A = self.visibility_relations_horizontal()

        #  Initialize Network object
        InteractingNetworks.__init__(self, A, directed=False,
                                     silence_level=silence_level)

    def __str__(self):
        """
        Returns a string representation.
        """
        return 'VisibilityGraph: time series shape %s.\n%s' % (
            self.time_series.shape, InteractingNetworks.__str__(self))

    #
    #  Visibility methods
    #

    def visibility_relations(self):
        """
        TODO
        """
        if self.silence_level <= 1:
            print "Calculating visibility relations..."

        #  Prepare
        x = self.time_series
        t = self.timings
        N = len(self.time_series)
        A = np.zeros((N, N), dtype="int8")

        if self.missing_values:
            mv_indices = self.missing_value_indices

            code = r"""
            int i,j,k;
            float test;

            for (i = 0; i < N - 2; i++) {
                for (j = i + 2; j < N; j++) {
                    k = i + 1;

                    test = (x(j) - x(i)) / (t(j) - t(i));

                    while (!mv_indices(k)
                           && (x(k) - x(i)) / (t(k) - t(i)) < test && k < j) {
                        k++;
                    }

                    if (k == j)
                        A(i,j) = A(j,i) = 1;
                }
            }

            //  Add trivial connections of subsequent observations
            //  in time series
            for (i = 0; i < N - 1; i++) {
                if (!mv_indices(i) && !mv_indices(i+1))
                    A(i,i+1) = A(i+1,i) = 1;
            }
            """
            args = ['x', 't', 'N', 'A', 'mv_indices']

        else:
            code = r"""
            int i,j,k;
            float test;

            for (i = 0; i < N - 2; i++) {
                for (j = i + 2; j < N; j++) {
                    k = i + 1;

                    test = (x(j) - x(i)) / (t(j) - t(i));

                    while ((x(k) - x(i)) / (t(k) - t(i)) < test && k < j)
                        k++;

                    if (k == j)
                        A(i,j) = A(j,i) = 1;
                }
            }

            //  Add trivial connections of subsequent observations
            //  in time series
            for (i = 0; i < N - 1; i++)
                A(i,i+1) = A(i+1,i) = 1;
            """
            args = ['x', 't', 'N', 'A']

        weave_inline(locals(), code, args)
        return A

    def visibility_relations_horizontal(self):
        """
        TODO
        """
        if self.silence_level <= 1:
            print "Calculating horizontal visibility relations..."

        #  Prepare
        x = self.time_series
        t = self.timings
        N = len(self.time_series)
        A = np.zeros((N, N), dtype="int8")

        code = r"""
            int i,j,k;
            float minimum;

            for (i = 0; i < N - 2; i++) {
                for (j = i + 2; j < N; j++) {
                    k = i + 1;
                    minimum = fmin(x(i), x(j));

                    while (x(k) < minimum && k < j)
                        k++;

                    if (k == j)
                        A(i,j) = A(j,i) = 1;
                }
            }

            //  Add trivial connections of subsequent observations
            //  in time series
            for (i = 0; i < N - 1; i++)
                A(i,i+1) = A(i+1,i) = 1;
        """
        weave_inline(locals(), code, ['x', 't', 'N', 'A'])
        return A

    #
    #  Specific measures for visibility graphs
    #

    def retarded_degree(self):
        """Return number of neighbors in the past of a node."""
        #  Prepare
        retarded_degree = np.zeros(self.N)
        A = self.adjacency

        for i in xrange(self.N):
            retarded_degree[i] = A[i, :i].sum()

        return retarded_degree

    def advanced_degree(self):
        """Return number of neighbors in the future of a node."""
        #  Prepare
        advanced_degree = np.zeros(self.N)
        A = self.adjacency

        for i in xrange(self.N):
            advanced_degree[i] = A[i, i:].sum()

        return advanced_degree

    def retarded_local_clustering(self):
        """
        Return probability that two neighbors of a node in its past are
        connected.
        """
        #  Prepare
        retarded_clustering = np.zeros(self.N)

        #  Get full adjacency matrix
        A = self.adjacency
        #  Get number of nodes
        N = self.N

        #  Get left degree
        retarded_degree = self.retarded_degree()
        #  Prepare normalization factor
        norm = retarded_degree * (retarded_degree - 1) / 2.

        code = """
        long counter;

        //  Loop over all nodes
        for (int i = 2; i < N; i++) {
            //  Check if i has right degree larger than 1
            if (norm(i) != 0) {
                //  Reset counter
                counter = 0;

                //  Loop over unique pairs of nodes in the past of i
                for (int j = 0; j < i; j++) {
                    for (int k = 0; k < j; k++) {
                        if (A(i,j) == 1 && A(j,k) == 1
                            && A(k,i) == 1) {
                            counter++;
                        }
                    }
                }
                retarded_clustering(i) = counter / norm(i);
            }
        }
        """
        weave_inline(locals(), code, ['N', 'A', 'norm', 'retarded_clustering'])
        return retarded_clustering

    def advanced_local_clustering(self):
        """
        Return probability that two neighbors of a node in its future are
        connected.
        """
        #  Prepare
        advanced_clustering = np.zeros(self.N)

        #  Get full adjacency matrix
        A = self.adjacency
        #  Get number of nodes
        N = self.N

        #  Get right degree
        advanced_degree = self.advanced_degree()
        #  Prepare normalization factor
        norm = advanced_degree * (advanced_degree - 1) / 2.

        code = """
        long counter;

        //  Loop over all nodes
        for (int i = 0; i < N - 2; i++) {
            //  Check if i has right degree larger than 1
            if (norm(i) != 0) {
                //  Reset counter
                counter = 0;

                //  Loop over unique pairs of nodes in the future of i
                for (int j = i + 1; j < N; j++) {
                    for (int k = i + 1; k < j; k++) {
                        if (A(i,j) == 1 && A(j,k) == 1
                            && A(k,i) == 1) {
                            counter++;
                        }
                    }
                }
                advanced_clustering(i) = counter / norm(i);
            }
        }
        """
        weave_inline(locals(), code, ['N', 'A', 'norm', 'advanced_clustering'])
        return advanced_clustering

    def retarded_closeness(self):
        """Return average path length to nodes in the past of a node."""
        #  Prepare
        retarded_closeness = np.zeros(self.N)
        path_lengths = self.path_lengths()

        for i in xrange(self.N):
            retarded_closeness[i] = path_lengths[i, :i].mean() ** (-1)

        return retarded_closeness

    def advanced_closeness(self):
        """Return average path length to nodes in the future of a node."""
        #  Prepare
        advanced_closeness = np.zeros(self.N)
        path_lengths = self.path_lengths()

        for i in xrange(self.N):
            advanced_closeness[i] = path_lengths[i, i+1:].mean() ** (-1)

        return advanced_closeness

    def retarded_betweenness(self):
        """
        Return betweenness of a node with respect to all pairs of nodes in its
        past.
        """
        #  Prepare
        retarded_betweenness = np.zeros(self.N)

        for i in xrange(self.N):
            retarded_indices = np.arange(i)
            retarded_betweenness[i] = self.nsi_betweenness(
                sources=retarded_indices, targets=retarded_indices)[i]

        return retarded_betweenness

    def advanced_betweenness(self):
        """
        Return betweenness of a node with respect to all pairs of nodes in its
        future.
        """
        #  Prepare
        advanced_betweenness = np.zeros(self.N)

        for i in xrange(self.N):
            advanced_indices = np.arange(i+1, self.N)
            advanced_betweenness[i] = self.nsi_betweenness(
                sources=advanced_indices, targets=advanced_indices)[i]

        return advanced_betweenness

    def trans_betweenness(self):
        """
        Return betweenness of a node with respect to all pairs of nodes
        with one node the past and one node in the future, respectively.
        """
        #  Prepare
        trans_betweenness = np.zeros(self.N)

        for i in xrange(self.N):
            retarded_indices = np.arange(i)
            advanced_indices = np.arange(i+1, self.N)
            trans_betweenness[i] = self.nsi_betweenness(
                sources=retarded_indices, targets=advanced_indices)[i]

        return trans_betweenness

    #
    #  Measures corrected for boundary effects
    #

    def boundary_corrected_degree(self):
        """Return a weighted degree corrected for trivial boundary effects."""
        #  Prepare
        N_past = np.arange(self.N)
        N_future = N_past[::-1]

        cdegree = (self.retarded_degree() * N_past
                   + self.advanced_degree() * N_future) / float(self.N - 1)

        return cdegree

    def boundary_corrected_closeness(self):
        """
        Return a weighted closeness corrected for trivial boundary effects.
        """
        #  Prepare
        N_past = np.arange(self.N)
        N_future = N_past[::-1]

        ccloseness = (self.N - 1) * (self.retarded_closeness() / N_past +
                                     self.advanced_closeness() / N_future)

        return ccloseness
