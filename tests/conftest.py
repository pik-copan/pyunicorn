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

from pathlib import Path

import requests
import pytest


@pytest.fixture(scope="session",
                params=["supremum", "euclidean", "manhattan"])
def metric(request) -> str:
    '''
    A fixture for creating parametrized fixtures of classes that have a
    `metric` argument, as in `RecurrencePlot` and its child classes.
    '''
    return request.param


@pytest.fixture(scope="session")
def reanalysis_data() -> Path:
    """
    Locate, and potentially download, a small NOAA dataset. Currently used in:
    - `tests/test_climate/test_map_plot.py`
    - `docs/source/examples/tutorials/ClimateNetworks.ipynb`
    """
    service = "https://psl.noaa.gov/repository/entry/get"
    data_name = "air.mon.mean.nc"
    query = (
        "entryid=synth%3Ae570c8f9-ec09-4e89-93b4-"
        "babd5651e7a9%3AL25jZXAucmVhbmFseXNpcy5kZXJpdm"
        "VkL3N1cmZhY2UvYWlyLm1vbi5tZWFuLm5j")
    url = f"{service}/{data_name}?{query}"

    data_dir = Path("./docs/source/examples/tutorials/data")
    data_file = data_dir / data_name

    if not data_dir.is_dir():
        data_dir.mkdir(parents=True)
    if not data_file.exists():
        res = requests.get(url, timeout=(30, 30))
        res.raise_for_status()
        with open(data_file, 'wb') as f:
            f.write(res.content)

    return data_file
