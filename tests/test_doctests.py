#! /usr/bin/env python2

# This file is part of pyunicorn.
# Copyright (C) 2008--2015 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)

from os import walk
from os.path import normpath, join, dirname, sep
from importlib import import_module
from sys import exit
import argparse
import doctest

import numpy as np


ignored_folders = ['ropeproject', 'progressbar']
ignored_modules = ['__init__', 'progressbar', 'navigator']


def r(obj):
    """
    Round numbers, arrays or iterables thereof for doctests.
    """
    if type(obj) in [np.ndarray, np.matrix]:
        if obj.dtype.kind == 'f':
            return np.around(obj.astype(np.float128),
                             decimals=4).astype(np.float)
        elif obj.dtype.kind == 'i':
            return obj.astype(np.int)
    elif type(obj) is list:
        return map(r, obj)
    elif type(obj) is tuple:
        return tuple(map(r, obj))
    elif type(obj) in [float, np.float32, np.float64, np.float128]:
        return np.float(np.around(np.float128(obj), decimals=4))
    elif type(obj) in [int, np.int8, np.int16]:
        return int(obj)
    else:
        return obj


def rr(obj):
    """
    Force arrays in stubborn scientific notation into a few digits.
    """
    print np.vectorize('%.4g'.__mod__)(r(obj))


doctest_opts = {
    'extraglobs': {'r': r, 'rr': rr},
    'optionflags': doctest.NORMALIZE_WHITESPACE
}


def load_tests(loader, tests, ignore):
    """
    Export all doctests when scanned by unittest.
    """
    lib = normpath(join(dirname(__file__), '..', 'pyunicorn'))
    focus = lambda cand, ignored: all(cand.find(s) < 0 for s in ignored)
    modules = []
    for d, _, fs in walk(lib):
        if focus(d, ignored_folders):
            modules.extend(join(d, f)[:-3].replace(sep, '.') for f in fs
                           if f.endswith('.py') and focus(f, ignored_modules))
    for module in modules:
        tests.addTests(doctest.DocTestSuite(module, **doctest_opts))
    return tests


def main():
    """
    Run doctests in modules given as arguments when invoked by shell.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('modules', metavar='mod', nargs='+',
                        help='modules to be doctested (pyunicorn.*)')
    for mod in parser.parse_args().modules:
        try:
            module = import_module('pyunicorn.' + mod)
        except ImportError:
            print "Failed to import module: ", mod
            exit(1)
        doctest.testmod(module, **doctest_opts)


if __name__ == "__main__":
    main()
