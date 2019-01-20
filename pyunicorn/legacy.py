import numpy as np


def new_numpy_version():
    return np.__version__ >= '1.14'


def set_legacy_mode():
    args = {}
    if new_numpy_version():
        args['legacy'] = '1.13'
    np.set_printoptions(**args)
