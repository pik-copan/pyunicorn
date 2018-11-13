import numpy as np


def new_numpy_version():
    return '1.15' in np.__version__ or '1.14' in np.__version__


def set_legacy_mode():
    args = {}
    if new_numpy_version() is True:
        args['legacy'] = '1.13'
    np.set_printoptions(**args)
