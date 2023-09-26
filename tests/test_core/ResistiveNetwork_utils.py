# Copyright (C) 2014 SWIPO Project
#
# Authors (this file):
#   Stefan Schinkel <stefan.schinkel@gmail.com>

"""
Utils needed for Unit Tests in resistive networks
"""
import numpy as np


def makeNW(idI, idJ, val=None):
    """Construct NW from edges
    """

    # idI and IdJ must be lists!
    idI = list(idI)
    idJ = list(idJ)

    # get N
    N = max(list(idI)+list(idJ))

    # empty matrix
    nw = np.zeros((N+1, N+1))

    # fill values or ones
    if val is None:
        val = [1] * len(idI)
    elif len(val) == 1:
        val = val*len(idI)

    for i, j, v in zip(idI, idJ, val):
        nw[i, j] = v
        nw[j, i] = v

    return nw


def parallelCopy(nw, a, b):
    """ copy a graph of N nodes with a given adjacency
    matrix and attach the copy linking node them at node
    a and b (parallel circuit)
    """

    if sum(np.diag(nw)) > 0:
        print("Graph has selfloops")
        print("doesn't work yet")
        raise NotImplementedError

        # at this point, have to copy the diagonal
        # and fill that to the copy
        # np.fill_diagonal(nw,0)

    # get size of NW
    N = len(nw)

    # get the current adjacency and values
    # of the upper triangle only (has to be symmetric)
    i0, j0 = np.nonzero(np.triu(nw))
    val = []
    for ii, jj in zip(i0, j0):
        val.append(nw[ii, jj])

    # increment i indeces and remap
    ti = i0+N
    ti[ti == a+N] = a
    ii = np.append(i0, ti)

    # same for j
    tj = j0+N
    tj[tj == b+N] = b
    jj = np.append(j0, tj)

    # compute return value and adjust
    # the values for direct links
    nw2 = makeNW(ii, jj, val*2)
    if nw[a, b] != 0:
        nw2[a, b] = nw[a, b]/2
        nw2[b, a] = nw[a, b]/2

    return nw2


def serialCopy(nw):
    """ Copy a graph and attach a copy to
    the end (serial circuit)
    """

    # get size of NW
    N = len(nw)

    # get the current adjacency and values
    # of the upper triangle only (has to be symmetric)
    i0, j0 = np.nonzero(np.triu(nw))
    val = []
    for ii, jj in zip(i0, j0):
        val.append(nw[ii, jj])

    # increment indeces by N and append
    ti = i0+N-1
    ii = np.append(i0, ti)

    # same for j
    tj = j0+N-1
    jj = np.append(j0, tj)

    return makeNW(ii, jj, val*2)


def nx2nw(G):
    """Convert networkx Graph to a plain matrix
    (ndarray)
    """
    i = []
    j = []
    for edge in G.edges():
        i.append(edge[0])
        j.append(edge[1])

    return makeNW(i, j)
