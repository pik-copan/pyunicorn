#! /usr/bin/env python2

# This file is part of pyunicorn.
# Copyright (C) 2008--2015 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)

"""
When scanned by unittest, export all pyunicorn doctests.
When invoked by shell, run doctests in pyunicorn submodules given as arguments
(must be globally importable).
"""

from os import walk
from os.path import normpath, join, dirname, sep, splitext
from importlib import import_module
import argparse
import doctest

import numpy as np

ignored_folders = ['ropeproject', 'progressbar']
ignored_modules = ['__init__', 'progressbar', 'navigator']


def r(obj):
    """
    Round numbers, arrays or iterables thereof for doctests.
    """
    if isinstance(obj, (np.ndarray, np.matrix)):
        if obj.dtype.kind == 'f':
            rounded = np.around(obj.astype(np.float128),
                                decimals=4).astype(np.float)
        elif obj.dtype.kind == 'i':
            rounded = obj.astype(np.int)
    elif isinstance(obj, list):
        rounded = map(r, obj)
    elif isinstance(obj, tuple):
        rounded = tuple(map(r, obj))
    elif isinstance(obj, (float, np.float32, np.float64, np.float128)):
        rounded = np.float(np.around(np.float128(obj), decimals=4))
    elif isinstance(obj, (int, np.int8, np.int16)):
        rounded = int(obj)
    else:
        rounded = obj
    return rounded


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
            modules.extend(join(d, splitext(f)[0]).replace(sep, '.')
                           for f in fs if splitext(f)[1] in ['.py', '.pyx']
                           and focus(f, ignored_modules))
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
            doctest.testmod(module, **doctest_opts)
        except ImportError:
            print "Failed to import module: ", mod
            exit(1)


if __name__ == "__main__":
    main()
