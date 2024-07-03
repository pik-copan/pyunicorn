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
Provides class for the analysis of dynamical systems and time series based
on event synchronization and event coincidence analysis
"""

# array object and fast numerics
import numpy as np

from ..eventseries import EventSeries
from .climate_network import ClimateNetwork
from .climate_data import ClimateData
from ..core import Data

#
#  Class definitions
#


class EventSeriesClimateNetwork(EventSeries, ClimateNetwork):
    """
    Class EventSeriesClimateNetwork for generating and quantitatively
    analyzing event synchronisation and event coincidence analysis networks

    References: [Boers2014]_.
    """

    #
    #  Internal methods

    def __init__(self, data, method='ES', p_value=None, **kwargs):
        r"""
        Initialize an instance of EventSeriesClimateNetwork.

        For other applications of event series networks please use
        the EventSeries class together with the Network class.

        :type data: :class:`..climate.ClimateData`
        :arg data: The climate data used for network construction.
        :type method: ``str {'ES', 'ECA', 'ES_pval', 'ECA_pval'}``
        :arg method: determines if ES, ECA, or the p-values of one of the
                     methods should be used for network reconstruction.
        :type p_value: ``float in [0,1]``
        :arg p_value: determines the p-value threshold for network
                      reconstruction. ES/ECA scores of event time series pairs
                      with p-value higher than threshold are set to zero
                      leading to missing link in climate network.
                      Default: None. No p-value thresholding.
        :arg \**kwargs:
            optional keyword arguments to specify parent classes' behavior,
            see below for all options.

        :Keyword Arguments:
            * *taumax* (``float``) --
              maximum time difference between two events to be
              considered synchronous. Caution: For ES, the default is
              np.inf because of the intrinsic dynamic coincidence
              interval in ES. For ECA, taumax is a parameter of the
              method that needs to be defined.
            * *lag* (``float``) --
              extra time lag between the event series.
            * *symmetrization* (
              ``str {'directed', 'symmetric', 'antisym',
              'mean', 'max', 'min'}`` for ES,
              str {'directed', 'mean', 'max', 'min'}`` for ECA
              ) --
              determines if and if true, which symmetrization
              should be used for the ES/ECA score matrix.
            * *window_type* (
              ``str {'retarded', 'advanced', 'symmetric'}``
              ) --
              Only for ECA. Determines if precursor coincidence
              rate ('advanced'), trigger coincidence rate
              ('retarded') or a general coincidence rate with the
              symmetric interval [-taumax, taumax] are computed
              ('symmetric'). Default: 'symmetric'.
            * *threshold_method* (
              ``str 'quantile' or 'value' or
              1D numpy array of 'quantile' or 'value'``
              ) --
              specifies the method for generating a binary
              event matrix from an array of continuous time
              series. Default: None.
            * *threshold_values* (``1D Numpy array or float``) --
              quantile or real number determining threshold
              for each variable. Default: None.
            * *threshold_types* (
              ``str 'above' or 'below' or 1D list
              of strings of 'above' or 'below'``
              ) --
              determines for each variable if event is below
              or above threshold.
            * *non_local* (``bool``) --
              determines whether links between spatially close
              nodes should be suppressed.
            * *node_weight_type* (``str``) --
              The type of geographical node weight to be
              used.
            * *arg silence_level* (``int``) --
              The inverse level of verbosity of the object.
        """

        # extract ES and CN related optional keyword arguments from **kwargs
        ES_kwargs = {
            "taumax": kwargs.get("taumax", np.inf),
            "lag": kwargs.get("lag", 0.0),
            "threshold_method": kwargs.get("threshold_method", None),
            "threshold_values": kwargs.get("threshold_values", None),
            "threshold_types": kwargs.get("threshold_types", None)
        }

        ES_analysis_kwargs = {
            "symmetrization": kwargs.get("symmetrization", 'directed'),
            "window_type": kwargs.get("window_type", 'symmetric')
        }

        ES_significance_kwargs = {
            "surrogate": kwargs.get("surrogate", 'shuffle'),
            "n_surr": kwargs.get("n_surr", 1000),
            "symmetrization": kwargs.get("symmetrization", 'directed'),
            "window_type": kwargs.get("window_type", 'symmetric')
        }

        CN_kwargs = {
            "non_local": kwargs.get("non_local", False),
            "node_weight_type": kwargs.get("node_weight_type", "surface"),
            "silence_level": kwargs.get("silence_level", 0)
        }

        method_types = ['ES', 'ECA', 'ES_pval', 'ECA_pval']
        if method not in method_types:
            raise IOError(f"Method input must be: "
                          f"{method_types[0]}, {method_types[1]},"
                          f"{method_types[2]}, or {method_types[3]}!")

        self.__method = method
        self.__p_value = p_value

        self.__symmetry = kwargs.get("symmetrization", 'directed')
        self.directed = self.__symmetry == "directed"

        # Construct an EventSeries object with the chosen parameters
        EventSeries.__init__(self, data.observable(), **ES_kwargs)

        # Compute matrix for link weights of ClimateNetwork from event
        # synchronization or event coincidence analysis with chosen symmetry
        # option
        measure_matrix = []

        # If standard ES/ECA measure is chosen, calculate pairwise ES/ECA
        # scores
        if self.__method in ['ES', 'ECA']:
            measure_matrix = \
                self.event_series_analysis(method=self.__method,
                                           **ES_analysis_kwargs)

            # Check if a p-value is chosen, then:
            # Set all coupling strengths in the measure matrix with
            # higher p-value to zero, leading to a missing edge in the climate
            # network
            if self.__p_value is not None:

                if self.__p_value > 1.0 or self.__p_value < 0.0:
                    raise IOError("'p_value' must lie in the unit interval!")

                significance_matrix = \
                    self.event_analysis_significance(
                        method=self.__method, **ES_significance_kwargs)

                for i in range(self.__N):
                    for j in range(self.__N):
                        if significance_matrix[i][j] < 1.0 - p_value:
                            measure_matrix[i][j] = 0.0

        elif self.__method in ['ES_pval', 'ECA_pval']:
            measure_matrix = \
                self.event_analysis_significance(
                    method=self.__method, **ES_significance_kwargs)

        ClimateNetwork.__init__(self, grid=data.grid,
                                similarity_measure=measure_matrix,
                                threshold=0, directed=self.directed,
                                **CN_kwargs)

    def __str__(self):
        """
        Return a string representation of EventSeriesClimateNetwork.

        **Example:**

        >>> data = EventSeriesClimateNetwork.SmallTestData()
        >>> print(EventSeriesClimateNetwork(data, taumax=16.0,
        >>>       threshold_method='quantile', threshold_value=0.8,
        >>>       threshold_types='above'))
        Extracting network adjacency matrix by thresholding...
        Setting area weights according to type surface ...
        Setting area weights according to type surface ...
        EventSeriesClimateNetwork:
        EventSeries: 6 variables, 10 timesteps, __taumax: 16.0, lag: 0.0
        ClimateNetwork:
        GeoNetwork:
        Network: directed, 6 nodes, 0 links, link density 0.000.
        Geographical boundaries:
                 time     lat     lon
           min    0.0    0.00    2.50
           max    9.0   25.00   15.00
        Threshold: 0
        Local connections filtered out: False
        Type of event series measure to construct
        the network: directedES
        """
        text = ("EventSeriesClimateNetwork:\n%s\n%s\n"
                "Type of event series measure to construct the network: "
                "%s%s")
        return text % (EventSeries.__str__(self),
                       ClimateNetwork.__str__(self), self.__symmetry,
                       self.__method)

    @staticmethod
    def SmallTestData():
        """
        Return test data set of 6 time series with 10 sampling points each.

        **Example:**

        >>> r(Data.SmallTestData().observable())
        array([[ 0.    ,  1.    ,  0.    , -1.    , -0.    ,  1.    ],
               [ 0.309 ,  0.9511, -0.309 , -0.9511,  0.309 ,  0.9511],
               [ 0.5878,  0.809 , -0.5878, -0.809 ,  0.5878,  0.809 ],
               [ 0.809 ,  0.5878, -0.809 , -0.5878,  0.809 ,  0.5878],
               [ 0.9511,  0.309 , -0.9511, -0.309 ,  0.9511,  0.309 ],
               [ 1.    ,  0.    , -1.    , -0.    ,  1.    ,  0.    ],
               [ 0.9511, -0.309 , -0.9511,  0.309 ,  0.9511, -0.309 ],
               [ 0.809 , -0.5878, -0.809 ,  0.5878,  0.809 , -0.5878],
               [ 0.5878, -0.809 , -0.5878,  0.809 ,  0.5878, -0.809 ],
               [ 0.309 , -0.9511, -0.309 ,  0.9511,  0.309 , -0.9511]])

        :rtype: ClimateData instance
        :return: a ClimateData instance for testing purposes.
        """
        data = Data.SmallTestData()

        return ClimateData(observable=data.observable(), grid=data.grid,
                           time_cycle=5, silence_level=2)
