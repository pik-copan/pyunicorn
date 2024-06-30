# Copyright (C) 2014 SWIPO Project
#
# Authors (this file):
#   Stefan Schinkel <stefan.schinkel@gmail.com>
"""
Provides sanity checks for basic for parallel and serial circiuts.
"""
import numpy as np
import networkx as nx


from pyunicorn import ResNetwork
from .ResistiveNetwork_utils import \
    makeNW, parallelCopy, serialCopy, nx2nw

debug = 0
""" Test for basic sanity, parallel and serial circiuts
"""


def testParallelTrivial():
    r""" Trivial parallel case:
    a)  0 --- 1 --- 2

        /---- 3 ---\
    b)  0 --- 1 --- 2

    c)  /---- 3 ---\
        0 --- 1 --- 2
        \____ 4 ___/

    ER(a) = 2*ER(b) = 3*ER(c)
    """
    nws = []
    # construct nw1
    idI, idJ = [0, 1], [1, 2]
    nws.append(makeNW(idI, idJ, [.1]))

    # construct nw2
    idI += [0, 3]
    idJ += [3, 2]
    nws.append(makeNW(idI, idJ, [.1]))

    # nw3
    idI += [0, 4]
    idJ += [4, 2]
    nws.append(makeNW(idI, idJ, [.1]))

    ER = []
    for nw in nws:
        rnw = ResNetwork(nw)
        ER.append(rnw.effective_resistance(0, 2))

    assert abs(ER[0]/2-ER[1]) < .1E-6
    assert abs(ER[0]/3-ER[2]) < .1E-6


def testParallelLessTrivial():
    """ Less Trivial Parallel Case:
            |--- 1 --- 0
        a) 2     |
            |--- 3 ----4

            |--- 1 --- 0 --- 5 --- |
        b) 2     |           |     7
            |--- 3 ----4 --- 6 --- |

                      |---- 8 ----------- |
                      |     |             |
                      |     |----------|  |
                      |                |  |
           |--- 1 --- 0 --- 5 --- |    |  |
        c) 2    |           |     7    |  9
           |--- 3 ----4 --- 6 --- |    |  |
                      |                |  |
                      |      ----------|  |
                      |      |            |
                      |---- 10 -----------|
    """
    nws = []

    idI = [0, 1, 1, 2, 3]
    idJ = [1, 2, 3, 3, 4]
    nws.append(makeNW(idI, idJ, [1]*len(idI)))

    idI.extend([0, 5, 5, 6, 6])
    idJ.extend([5, 6, 7, 7, 4])
    nws.append(makeNW(idI, idJ, [1]*len(idI)))

    idI.extend([0, 8, 8, 9, 10])
    idJ.extend([8, 9, 10, 10, 4])
    nws.append(makeNW(idI, idJ, [1]*len(idI)))

    ER = []
    for nw in nws:
        rnw = ResNetwork(nw)
        ER.append(rnw.effective_resistance(0, 4))
    #     Gs.append(nx.DiGraph(nw))
    # # showGraphs(Gs)
    # # s = ''
    # # for i,e in enumerate(ER):
    # #     s = s + "NW{:d} {:.3f}\t".format(i,e)
    # # print("Effective resistances (0,2)\n %s" % (s))

    assert abs(ER[0]/2-ER[1]) < .1E-6
    assert abs(ER[0]/3-ER[2]) < .1E-6

    # """ Less Trivial Parallel Case:
    #     /--- 1 --- 0
    # a) 2     |
    #     \--- 3 ----4

    #     /--- 1 --- 0 --- 5 --- \
    # b) 2     |           |      7
    #     \--- 3 ----4 --- 6 --- /

    #               / --- 8 ----------- \
    #               |                    \
    #    /--- 1 --- 0 --- 5 --- \         \
    # c) 2                       7        9
    #    \--- 3 ----4 --- 6 --- /         /
    #               |                    /
    #               \ --- 10 -----------/
    # """
    # nws =[]
    # #construct nw1

    # idI = [0,1,1,2,3]
    # idJ = [1,2,3,3,4]
    # val = [.1] * 5
    # nws.append(makeNW(idI,idJ,[.1]*len(idI))[0])

    # idI.extend([0,5,6,7])
    # idJ.extend([5,6,7,4])
    # val.extend( val * 6)
    # nws.append(makeNW(idI,idJ,[.1]*len(idI))[0])

    # idI.extend([0,8,9,10])
    # idJ.extend([8,9,10,4])
    # val.extend( val * 4)
    # nws.append(makeNW(idI,idJ,val)[0])

    # ER = []
    # for nw in nws:
    #     rnw = ResNetwork(nw)
    #     ER.append( rnw.effective_resistance(0,4))

    # s = ''
    # for i,e in enumerate(ER):
    #     s = s + "NW{:d} {:.3f}\t".format(i,e)
    # print("Effective resistances (0,2)\n %s" % (s))

    # assert abs(ER[0]/2-ER[1]) < .1E-6
    # assert abs(ER[0]/3-ER[2]) < .1E-6


def testParallelRandom():
    """ 50 random parallel cases
    """

    N = 10
    p = .7

    runs = 0
    while runs < 50:

        G = nx.fast_gnp_random_graph(N, p)
        a = 0
        b = G.number_of_nodes()-1

        try:
            nx.shortest_path(G, source=a, target=b)
        except RuntimeError:
            continue

        i, j = [], []
        for xx in G.edges():
            i.append(xx[0])
            j.append(xx[1])

        # %.1f values for resistance
        val = np.round(np.random.rand(len(i))*100)/10

        # and test
        nw1 = makeNW(i, j, val)
        nw2 = parallelCopy(nw1, a, b)
        ER1 = ResNetwork(nw1).effective_resistance(a, b)
        ER2 = ResNetwork(nw2).effective_resistance(a, b)

        # assertion
        assert (ER1/2-ER2) < 1E-6

        # increment runs
        runs += 1


def testSerialTrivial():
    """Trivial serial test case

    a) 0 --- 1 --- 2

    b) 0 --- 1 --- 2 --- 3 --- 4

    ER(a)/2 = ER(b)
    """

    # construct nw1
    idI = [0, 1]
    idJ = [1, 2]
    val = [1, 1]

    nw1 = np.zeros((3, 3))
    for i, j, v in zip(idI, idJ, val):
        nw1[i, j] = v
        nw1[j, i] = v

    # construct nw2
    idI = idI + [2, 3]
    idJ = idJ + [3, 4]
    val = val + [1, 1]

    nw2 = np.zeros((5, 5))
    for i, j, v in zip(idI, idJ, val):
        nw2[i, j] = v
        nw2[j, i] = v

    # init ResNetworks
    rnw1 = ResNetwork(nw1)
    rnw2 = ResNetwork(nw2)

    ER1 = rnw1.effective_resistance(0, 2)
    ER2 = rnw2.effective_resistance(0, 4)

    print("Effective resistances (0,2)")
    print(f"NW1 {ER1:.3f}\tNW2 {ER2:.3f}\t 2*NW1 = {(2*ER1):.3f}")

    assert (ER1*2-ER2) < 1E-6


def testSerialRandom():
    """ 50 Random serial test cases
    """

    N = 10
    p = .7
    runs = 50
    for _ in range(0, runs):

        # a random graph
        G = nx.fast_gnp_random_graph(N, p)
        try:
            nx.shortest_path(G, source=0, target=N-1)
        except RuntimeError:
            continue
        except nx.NetworkXNoPath:
            pass
        # convert to plain ndarray
        nw1 = nx2nw(G)

        # copy and join network
        nw2 = serialCopy(nw1)

        # compute effective resistance
        ER1 = ResNetwork(
            nw1, silence_level=3).effective_resistance(0, len(nw1)-1)
        ER2 = ResNetwork(
            nw2, silence_level=3).effective_resistance(0, len(nw2)-1)

        # assertion
        # print(ER1*2-ER2)
        assert (ER1*2-ER2) < 1E-6
