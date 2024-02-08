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
Generic consistency checks.
"""

from pyunicorn import __version__, core, climate, timeseries, funcnet


def test_version():
    """
    Valid version information.
    """
    assert isinstance(__version__, str)


def test_init_str():
    """
    Reasonable __init__ and __str__ for most classes.
    """
    for i in simple_instances():
        assert str(i).startswith(i.__class__.__name__)


def simple_instances():
    """
    Minimally initialized instances for most classes.
    """
    cd = climate.ClimateData.SmallTestData()
    ca = funcnet.CouplingAnalysis.test_data()[:50]
    ts = ca[:, 0]
    t1, t2, t3 = [{'threshold': t} for t in [0.2, (0.2, 0.2), (0.2, 0.2, 0.2)]]
    # es = 1-(np.random.rand(100, 10) > 0.4).astype(int)
    # ec = climate.EventSynchronizationClimateNetwork.SmallTestData()
    return [
        core.Network.SmallTestNetwork(),
        core.Grid.SmallTestGrid(),
        core.SpatialNetwork.SmallTestNetwork(),
        core.GeoGrid.SmallTestGrid(),
        core.GeoNetwork.SmallTestNetwork(),
        core.InteractingNetworks.SmallTestNetwork(),
        core.ResNetwork.SmallTestNetwork(),
        core.NetCDFDictionary(),
        cd,
        climate.ClimateNetwork.SmallTestNetwork(),
        climate.HavlinClimateNetwork(cd, 0, **t1),
        climate.HilbertClimateNetwork(cd, **t1),
        climate.TsonisClimateNetwork.SmallTestNetwork(),
        climate.PartialCorrelationClimateNetwork(cd, winter_only=False, **t1),
        climate.RainfallClimateNetwork(cd, **t1),
        climate.SpearmanClimateNetwork(cd, winter_only=False, **t1),
        climate.MutualInfoClimateNetwork(cd, winter_only=False, **t1),
        climate.CoupledTsonisClimateNetwork(cd, cd, **t1),
        # climate.EventSynchronizationClimateNetwork(
        #     ec, 0.8, 16, eventsynctype="directedES"),
        timeseries.RecurrencePlot(ts, **t1),
        timeseries.RecurrenceNetwork(ts, **t1),
        timeseries.JointRecurrencePlot(ts, ts, **t2),
        timeseries.JointRecurrenceNetwork(ts, ts, **t2),
        timeseries.CrossRecurrencePlot(ts, ts, **t1),
        timeseries.InterSystemRecurrenceNetwork(ts, ts, **t3),
        timeseries.Surrogates.SmallTestData(),
        timeseries.VisibilityGraph(ts),
        funcnet.CouplingAnalysis(ca),
        funcnet.CouplingAnalysisPurePython(ca),
        # funcnet.EventSynchronization(es, 16)
    ]
