#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2019 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
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
Created on 28.05.2010

@author: heitzig-j
"""

#
#  Imports
#

import os
import time

import numpy as np
from numpy import random
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap

from .. import nz_coords


class Navigator(object):
    '''
    An interactive network navigator. EXPERIMENTAL
    '''

    def __init__(self,
                 # the network to navigate
                 network=None,
                 # list mapping (super-)node id to parent id (>id)
                 parent=None,
                 # list of information loss in joining children of a supernode
                 loss=None,
                 label=None,
                 autolabels=True,
                 loc=None,
                 solutions=[6, 6**2, 6**3],
                 map=False,
                 degree=None,
                 clustering=None,
                 centrality=None,
                 betweenness=None,
                 distance=None,
                 link_betweenness=None
                 ):
        '''
        Constructor
        '''
        self.map = map
        if map:
            self.fixed = True
            self.bm = Basemap()
            self.bm.drawcoastlines()
            self.fig = mpl.pyplot.gcf()
            self.ax = mpl.pyplot.gca()
            self.max_radius = 90
        else:
            self.fixed = False
            self.fig = mpl.pyplot.figure()
            self.ax = mpl.pyplot.gca()
            self.ax.set_xlim(-10000, 10000)
            self.ax.set_ylim(-10000, 10000)
            self.max_radius = 1000

        # needed so that static event handlers gets self!
        self.fig.pyunicorn_navigator = self
        self._press_cid = self.fig.canvas.mpl_connect('button_press_event',
                                                      self.on_press)
        self._motion_cid = self.fig.canvas.mpl_connect('motion_notify_event',
                                                       self.on_motion)
        self.dragging = None
        self.lastclick = time.time()-1

        self.loss = loss
        self.label = label
        self.autolabels = autolabels
        self.loc = loc
        self.solutions = solutions
        self.dim = 2

        self.set_degree(degree)
        self.set_clustering(clustering)
        self.set_centrality(centrality)
        self.set_betweenness(betweenness)
        self.set_distance(distance)
        self.set_link_betweenness(link_betweenness)

        self.set_network(network)
        if parent is not None:
            self.set_tree(parent, loss)

    @staticmethod
    def on_press(event):
        if event.inaxes is None:
            return  # not left clicked
        self = event.inaxes.figure.pyunicorn_navigator
        self.dragging = None
        if event.button == 1:
            if time.time() - self.lastclick < 0.3:  # doubleclick
                self._process_expand_button_on(self.get_target(event))
                self.lastclick = time.time()-1
            else:
                self.lastclick = time.time()
        elif event.button == 3:
            self._process_collapse_button_on(self.get_target(event))

    @staticmethod
    def on_motion(event):
        if event.inaxes is None:
            return
        self = event.inaxes.figure.pyunicorn_navigator
        if self.fixed:
            return
        if event.button != 1:
            self.dragging = None
            return  # not dragging
        pos = np.array([[event.xdata, event.ydata]])
        if self.dragging is None:
            dist = np.sqrt(((
                self.position[self.is_shown, :] - pos)**2).sum(axis=1))
            cands = np.where(dist < self.radius[self.is_shown] + 500.0)[0]
            if cands.size == 0:
                return
            self.dragging = np.arange(
                self.S)[self.is_shown][cands[dist[cands].argmin()]]
        self._move(self.dragging, pos.flatten())
        # self._improve_step()

        # TODO: on hover, show controls!

    def get_target(self, event):
        pos = np.array([[event.xdata, event.ydata]])
        dist = np.sqrt(((
            self.position[self.is_shown, :] - pos)**2).sum(axis=1))
        cands = np.where(dist < self.max_radius)[0]
        if cands.size == 0:
            return None
        return np.arange(self.S)[self.is_shown][cands[dist[cands].argmin()]]

    def set_network(self, network):
        self.network = network
        if network is not None and "parent" in self.__dict__:
            self._process_network_and_tree()
        # TODO: adjust

    def set_tree(self, parent, loss=None):
        self.parent = parent.tolist()
        self.loss = loss
        S = self.S = len(parent)+1
        # expansion state of supernodes by id
        self.is_expanded = np.zeros(S).astype("bool")
        # whether supernode is shown
        self.is_shown = np.zeros(S).astype("bool")
        self.shown = set([])
        # position of (shown) supernodes
        self.position = np.zeros((S, self.dim))
        self.lines = [{} for i in range(S)]
        self.circles = [None for i in range(S)]
        self.polys = [[] for i in range(S)]
        self.children = [[] for i in range(S)]
        for i in range(S-1):
            if parent[i] >= 0:
                self.children[parent[i]].append(i)
        if self.network is not None:
            self._process_network_and_tree()

    def set_degree(self, k):
        self.k0 = k

    def set_clustering(self, C):
        self.C0 = C

    def set_centrality(self, CC):
        self.CC0 = CC

    def set_betweenness(self, B):
        self.B0 = B

    def set_distance(self, D):
        self.D0 = D

    def set_link_betweenness(self, LB):
        self.LB0 = LB

    def _process_network_and_tree(self):
        # determine weights and linked proportions:
        N = self.N = self.network.N
        S = self.S
        self.weight = np.zeros(self.S)  # weight of supernodes
        sum_k = np.zeros(S)
        sum_C = np.zeros(S)
        sum_CC = np.zeros(S)
        sum_B = np.zeros(S)
        self.weight[:self.N] = \
            self.network.node_weights / self.network.total_node_weight
        if self.k0 is None:
            sum_k[:N] = np.zeros(N)
        else:
            sum_k[:N] = self.k0
        sum_k *= self.weight
        if self.C0 is None:
            sum_C[:N] = np.zeros(N)
        else:
            sum_C[:N] = self.C0
        sum_C *= self.weight
        if self.CC0 is None:
            sum_CC[:N] = np.zeros(N)
        else:
            sum_CC[:N] = self.CC0
        sum_CC *= self.weight
        if self.B0 is None:
            sum_B[:N] = np.zeros(N)
        else:
            sum_B[:N] = self.B0
        sum_B *= self.weight
        self.linked_weight = np.zeros((S, S))  # csc_matrix((self.S,self.S))
        self.linked_weight[range(N), range(N)] = self.weight[:self.N]**2
        for i, j in nz_coords(self.network.sp_A):
            self.linked_weight[i, j] = self.weight[i] * self.weight[j]
        if self.D0 is not None:
            sum_D = np.zeros((S, S))
            sum_D[:N, :N] = self.D0 * (self.weight[:N].reshape((-1, 1)) *
                                       self.weight[:N].reshape((1, -1)))
            print(sum_D)
        else:
            sum_D = None
        if self.LB0 is not None:
            sum_LB = np.zeros((S, S))
            sum_LB[:N, :N] = self.LB0 * self.linked_weight[:N, :N]
        else:
            sum_LB = None
        self.nodes = [[i] for i in range(N)] + [[] for i in range(N, S)]
        # this relies on parents having higher indices:
        for i in range(self.N, self.S):
            C = self.children[i]
            self.weight[i] = self.weight[C].sum()
            sum_k[i] = sum_k[C].sum()
            sum_C[i] = sum_C[C].sum()
            sum_CC[i] = sum_CC[C].sum()
            sum_B[i] = sum_B[C].sum()
            lwi = np.zeros(S)
            for j in C:  # TODO!!
                self.nodes[i] += self.nodes[j]
                lwi += self.linked_weight[:, j]
            self.linked_weight[:, i] = self.linked_weight[i, :] = lwi
            if sum_D is not None:
                di = np.zeros(S)
                for j in C:
                    di += sum_D[:, j]
                sum_D[:, i] = sum_D[i, :] = di
            if sum_LB is not None:
                lbi = np.zeros(S)
                for j in C:
                    lbi += sum_LB[:, j]
                sum_LB[:, i] = sum_LB[i, :] = lbi
        self.k = sum_k / self.weight
        self.C = sum_C / self.weight
        self.CC = sum_CC / self.weight
        self.B = sum_B / self.weight
        fac = 1 / (self.weight.reshape((-1, 1)) * self.weight.reshape((1, -1)))
        self.linked_proportion = self.linked_weight * fac
        if sum_D is not None:
            self.D = sum_D * fac
            print(sum_D)
            print(self.D)
        if sum_LB is not None:
            self.LB = sum_LB * fac
        del fac
        uc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        lc = "abcdefghijklmnopqrstuvwxyz"
        labelparts = [[uc[i] for i in range(23)]
                      + ["X"+uc[i] for i in range(26)]
                      + ["Y"+uc[i] for i in range(26)]
                      + ["Z"+uc[i] for i in range(26)],
                      xrange(1, 100),
                      [lc[i] for i in range(23)]
                      + ["x"+lc[i] for i in range(26)]
                      + ["y"+lc[i] for i in range(26)]
                      + ["z"+lc[i] for i in range(26)],
                      xrange(1, 100),
                      [lc[i] for i in range(23)]
                      + ["x"+lc[i] for i in range(26)]
                      + ["y"+lc[i] for i in range(26)]
                      + ["z"+lc[i] for i in range(26)]]
        labelnullpart = "~~~~~"
        if self.autolabels:
            self.label = [str(S-i) for i in range(S)]
            self.label[S-1] = ""
            pa = [S-1]
            grandpa = np.zeros(S).astype("int") + S-1
            self.solution_nodes = [None for s in self.solutions]
            level = -1
            while len(pa) < np.max(self.solutions):
                c = max(pa)
                pa.remove(c)
                C = self.children[c]
                pa += C
                grandpa[C] = grandpa[c]
                if len(pa) in self.solutions:
                    level += 1
                    self.solution_nodes[level] = pa+[]
                    paas = (-self.weight[pa]).argsort()
                    inds = np.zeros(S).astype("int")
                    for i in range(len(pa)):
                        j = pa[paas[i]]
                        gp = grandpa[j]
                        if gp != j:
                            self.label[j] = self.label[gp] + \
                                str(labelparts[level][inds[gp]])
                            inds[gp] += 1
                        else:
                            self.label[j] = self.label[gp] + \
                                labelnullpart[level]
                        print(j, self.label[j])
                    grandpa[pa] = pa
                elif level >= 0:
                    for i in C:
                        self.label[i] = self.label[grandpa[i]] + "." + str(S-i)
                        print(i, self.label[i])
        elif self.label is None:
            self.label = [str(i) for i in range(S)]
        self.is_geo = "grid" in self.network.__dict__
        if self.is_geo:
            self.lat = np.zeros(S)
            self.lon = np.zeros(S)
            self.lat[:N] = self.network.grid.get_lat_sequence()
            self.lon[:N] = self.network.grid.get_lon_sequence()
        self.boundary = [[] for i in range(S)]
        self.shape = [[] for i in range(S)]
        self.fullshape = [[] for i in range(S)]
        self.representative = [None for i in range(S)]
        if self.is_geo:
            for i in range(S):
                if self.label != str(S-i):
                    self.boundary[i], self.shape[i], self.fullshape[i],\
                        self.representative[i] = \
                        self.network.get_boundary(self.nodes[i], gap=0.1)
                    self.representative[i].append((0, 0))
                    self.lat[i], self.lon[i] = self.representative[i][0]
        if self.map:
            for i in range(self.S):
                self.position[i, 0], self.position[i, 1] = \
                    self.bm(self.lon[i], self.lat[i])
            #  radius of (shown) supernodes
            self.radius = 0.2*self.max_radius * \
                np.sqrt(self.weight/self.weight.max())
        else:
            # radius of (shown) supernodes
            self.radius = self.max_radius * \
                np.sqrt(self.weight/self.weight.max())
        color = random.uniform(size=(S, 3))
        color = np.floor(
            64+191*color/color.sum(axis=1).reshape((S, 1))).astype("int")
        self.color = \
            ["#"+hex(color[i, 0]*256**2+color[i, 1]*256+color[i, 2])[2:8]
             for i in range(S)]
        # TODO: use given distances!
        self.preferred_distance = 2000 * (10 - 9*self.linked_weight)  # 2000
        # self._process_expand_button_on(self.S-1)

    def draw_field(self, values, filename=None):
        fig = mpl.pyplot.figure()
        # fig.set_size_inches(18,9)
        ax = fig.add_axes((0, 0, 1, 1))
        bm = Basemap(ax=ax)
        bm.drawcoastlines(ax=ax)
        # fig.savefig("test.png",
        #             bbox_inches=mpl.transforms.Bbox([[0,0],[18,9]]))
        # fig.savefig("test.png",
        #             bbox_inches=mpl.transforms.Bbox([[0,1.03125],
        #                                              [8.125,5.09375]]))
        vmin = np.min(values)
        vrange = np.max(values)-vmin
        for i in range(self.N):
            b = (values[i]-vmin)*2.0/vrange
            if b < 1:
                c = [b, b, 1.0]
            else:
                c = [1.0, 2.0-b, 2.0-b]
            for sh in self.fullshape[i]:
                sh = np.array(sh)
                x, y = bm(sh[:, 1], sh[:, 0])
                xy = np.zeros((x.size, 2))
                xy[:, 0] = x
                xy[:, 1] = y
                if x.max()-x.min() < 180:
                    ax.add_patch(mpl.patches.Polygon(
                        xy, fc=c, lw=0.0, fill=True, ec="none", aa=False))
                else:
                    x1 = 1.0*x
                    x1[np.where(x < 0)] += 360.0
                    xy[:, 0] = x1
                    ax.add_patch(mpl.patches.Polygon(
                        xy, fc=c, lw=0.0, fill=True, ec="none", aa=False))
                    x1 -= 360.0
                    xy1 = 1*xy
                    xy1[:, 0] = x1
                    ax.add_patch(mpl.patches.Polygon(
                        xy1, fc=c, lw=0.0, fill=True, ec="none", aa=False))
        fig.show()
        if filename is not None:
            # fig.savefig(filename, bbox_inches=mpl.transforms.Bbox(
            #     [[0.02, 1.05], [8.12, 5.09]]))
            fig.savefig(filename, bbox_inches=mpl.transforms.Bbox(
                [[0.02, 1.0], [7.99, 5.0]]))
        return fig

    def write_kml(self,
                  path="climate-network-analysis",
                  pathurl=
                  "http://localhost/~heitzig-j/climate-network-analysis",
                  title="Climate Network Analysis",
                  measures=None,
                  stats=None
                  ):
        # TODO: kmz, betweenness correctly, i.png, zip, doublettes
        f = file(path+"/open-this-in-googleearth.kml", "w")
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <NetworkLink>
    <name>&lt;i&gt;&lt;b&gt;"""+title+"""&lt;/b&gt;&lt;/i&gt;</name>
    <Snippet maxLines="1">(please expand folders as needed)</Snippet>
    <open>1</open>
    <Link>
<!--
*********************************************************************************
** if you unzipped the archive into your own (local) webserver,                **
** change the following to point to the file doc.kml inside it,                **
** for example:                                                                **
**    <href>http://localhost/~username/climate-network-analysis/doc.kml</href> **
*********************************************************************************
-->
      <href>"""+pathurl+"""/doc.kml</href>
    </Link>
  </NetworkLink>
</kml>""")
        f.close()
        f = file(path+"/standard.kml", "w")
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<NetworkLinkControl>
  <Update>
    <targetHref>doc.kml</targetHref>
    <Change>
      <Folder targetId="resolution"><visibility>1</visibility></Folder>
      <Folder targetId="overlays"><visibility>0</visibility></Folder>
      <Folder targetId="options"><visibility>1</visibility></Folder>
      <Folder targetId="nodes"><visibility>1</visibility></Folder>
      <Folder targetId="links"><visibility>1</visibility></Folder>
      <Folder targetId="labels"><visibility>1</visibility></Folder>
    </Change>
  </Update>
</NetworkLinkControl>
</kml>""")
        f.close()
        f = file(path+"/no-standard.kml", "w")
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<NetworkLinkControl>
  <Update>
    <targetHref>doc.kml</targetHref>
    <Change>
      <Folder targetId="standard"><visibility>0</visibility></Folder>
    </Change>
  </Update>
</NetworkLinkControl>
</kml>""")
        # TODO: tour
        f = file(path+"/doc.kml", "w")
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">
  <Document>
    <Style id="check">
      <ListStyle><listItemType>check</listItemType></ListStyle>
    </Style>
    <Style id="radio">
      <ListStyle><listItemType>radioFolder</listItemType></ListStyle>
    </Style>
    <Style id="hide">
      <ListStyle><listItemType>checkHideChildren</listItemType><ItemIcon/></ListStyle>
    </Style>
    <Style id="noicon">
      <ListStyle><ItemIcon/></ListStyle>
    </Style>
    <!--<gx:Tour id="intro">
      <name>&lt;a href=&quot;#intro;flyto&quot;&gt;Introductory tour&lt;/a&gt;</name>
      <gx:Playlist>
        <gx:AnimatedUpdate>
          <Update>
          </Update>
        </gx:AnimatedUpdate>
      </gx:Playlist>
    </gx:Tour>-->
    <Folder id="standard">
      <name>&lt;b&gt;CHECK THIS FOR STANDARD SETTINGS&lt;/b&gt;</name>
      <visibility>0</visibility><styleUrl>#hide</styleUrl>
      <NetworkLink><Link><href>standard.kml</href></Link></NetworkLink>
    </Folder>
    <Folder id="resolution">
      <name>&lt;b&gt;Network resolution:&lt;/b&gt;</name>
      <open>1</open><visibility>1</visibility><styleUrl>#radio</styleUrl>
      <Folder>
        <name>Hierarchy of regions (with slider)...</name>
        <visibility>0</visibility><styleUrl>#radio</styleUrl>
        <Folder>
          <name>&lt;i&gt;show regions only&lt;/i&gt;</name>
          <visibility>0</visibility><styleUrl>#hide</styleUrl>
          <NetworkLink><Link><href>links-off.kml</href></Link></NetworkLink>
          <NetworkLink><Link><href>hierarchy-regions.kml</href></Link></NetworkLink>
        </Folder>
        <Folder>
          <name>&lt;i&gt;regions and links&lt;/i&gt;</name>
          <visibility>0</visibility><styleUrl>#hide</styleUrl>
          <NetworkLink><Link><href>no-standard.kml</href></Link></NetworkLink>
          <NetworkLink><Link><href>hierarchy-regions.kml</href></Link></NetworkLink>
          <NetworkLink><Link><href>hierarchy-links.kml</href></Link></NetworkLink>
        </Folder>
        <Folder>
          <name>&lt;i&gt;links only&lt;/i&gt;</name>
          <visibility>0</visibility><styleUrl>#hide</styleUrl>
          <NetworkLink><Link><href>no-standard.kml</href></Link></NetworkLink>
          <NetworkLink><Link><href>regions-off.kml</href></Link></NetworkLink>
          <NetworkLink><Link><href>hierarchy-links.kml</href></Link></NetworkLink>
        </Folder>
      </Folder>""")
        for level in range(len(self.solutions)):
            l = self.solutions[level]
            f.write("""
      <Folder>
        <name>Partition into """+str(l)+""" regions...</name>
        <visibility>0</visibility><styleUrl>#radio</styleUrl>
        <Folder>
          <name>&lt;i&gt;show regions only&lt;/i&gt;</name>
          <visibility>0</visibility><styleUrl>#hide</styleUrl>
          <NetworkLink><Link><href>no-standard.kml</href></Link></NetworkLink>
          <NetworkLink><Link><href>links-off.kml</href></Link></NetworkLink>
          <NetworkLink><Link><href>partition"""+str(l)+"""-regions.kml</href></Link></NetworkLink>
        </Folder>
        <Folder>
          <name>&lt;i&gt;regions and links&lt;/i&gt;</name>
          <visibility>0</visibility><styleUrl>#hide</styleUrl>
          <NetworkLink><Link><href>no-standard.kml</href></Link></NetworkLink>
          <NetworkLink><Link><href>partition"""+str(l)+"""-regions.kml</href></Link></NetworkLink>
          <NetworkLink><Link><href>partition"""+str(l)+"""-links.kml</href></Link></NetworkLink>
        </Folder>
        <Folder>
          <name>&lt;i&gt;links only&lt;/i&gt;</name>
          <visibility>0</visibility><styleUrl>#hide</styleUrl>
          <NetworkLink><Link><href>no-standard.kml</href></Link></NetworkLink>
          <NetworkLink><Link><href>regions-off.kml</href></Link></NetworkLink>
          <NetworkLink><Link><href>partition"""+str(l)+"""-links.kml</href></Link></NetworkLink>
        </Folder>
      </Folder>""")
        f.write("""
    </Folder>
    <Folder id="overlays">
      <name>&lt;b&gt;Color-coded maps:&lt;/b&gt;</name>
      <open>1</open><visibility>0</visibility><styleUrl>#radio</styleUrl>""")
        f.write("""
      <Folder>
        <name>Network measures...</name>
        <open>0</open><visibility>0</visibility><styleUrl>#radio</styleUrl>""")
        if measures is not None and len(measures) > 0:
            for i in range(len(measures)):
                self.draw_field(measures[i][1],
                                filename=path+"/measure"+str(i)+".png")
                f.write("""
        <GroundOverlay>
          <name>&lt;i&gt;"""+measures[i][0]+"""&lt;/i&gt;</name>
          <visibility>0</visibility><styleUrl>#noicon</styleUrl>
          <drawOrder>1</drawOrder>
          <Icon><href>measure"""+str(i)+""".png</href></Icon>
          <Region>
            <LatLonAltBox><north>90</north><south>-90</south><east>180.1</east><west>-180.1</west><minAltitude>0</minAltitude><maxAltitude>0</maxAltitude></LatLonAltBox>
            <Lod><minLodPixels>5</minLodPixels><maxLodPixels>12000</maxLodPixels><minFadeExtent>5</minFadeExtent><maxFadeExtent>11000</maxFadeExtent></Lod>
          </Region>
          <LatLonBox><north>90</north><south>-90</south><east>-180.1</east><west>180.1</west></LatLonBox>
        </GroundOverlay>""")
            f.write("""
      </Folder>""")
        if stats is not None and len(stats) > 0:
            f.write("""
      <Folder>
        <name>Underlying data...</name>
        <open>0</open><visibility>0</visibility><styleUrl>#radio</styleUrl>""")
            for i in range(len(stats)):
                f.write("""
        <GroundOverlay>
          <name>&lt;i&gt;l"""+stats[i][0]+"""&lt;/i&gt;</name>
          <visibility>0</visibility><styleUrl>#noicon</styleUrl>
          <drawOrder>1</drawOrder>
          <Icon><href>stat"""+str(0)+""".png</href></Icon>
          <Region>
            <LatLonAltBox><north>90</north><south>-90</south><east>180.1</east><west>-180.1</west><minAltitude>0</minAltitude><maxAltitude>0</maxAltitude></LatLonAltBox>
            <Lod><minLodPixels>5</minLodPixels><maxLodPixels>12000</maxLodPixels><minFadeExtent>5</minFadeExtent><maxFadeExtent>11000</maxFadeExtent></Lod>
          </Region>
          <LatLonBox><north>90</north><south>-90</south><east>-180</east><west>180</west></LatLonBox>
        </GroundOverlay>""")
            f.write("""
      </Folder>""")
        f.write("""
    </Folder>
    <Folder id="options">
      <name>&lt;b&gt;Options:&lt;/b&gt;</name>
      <open>1</open><visibility>0</visibility>
      <Folder id="nodes">
        <name>Node bullets...</name>
        <visibility>0</visibility><styleUrl>#radio</styleUrl>""")
        if self.B0 is not None:
            f.write("""
        <NetworkLink>
          <name>&lt;i&gt;colored by centrality and betweenness&lt;/i&gt;</name>
          <visibility>0</visibility><styleUrl>#noicon</styleUrl>
          <Link><href>nodes-betweenness.kml</href></Link>
        </NetworkLink>
        <NetworkLink>
          <name>&lt;i&gt;yellow&lt;/i&gt;</name>
          <visibility>0</visibility><styleUrl>#noicon</styleUrl>""")
        else:
            f.write("""
        <NetworkLink>
          <name>&lt;i&gt;yellow&lt;/i&gt;</name>
          <visibility>0</visibility><styleUrl>#noicon</styleUrl>""")
        f.write("""
          <Link><href>nodes-yellow.kml</href></Link>
        </NetworkLink>
        <NetworkLink>
          <name>&lt;i&gt;black&lt;/i&gt;</name>
          <visibility>0</visibility><styleUrl>#noicon</styleUrl>
          <Link><href>nodes-black.kml</href></Link>
        </NetworkLink>
        <NetworkLink>
          <name>&lt;i&gt;off&lt;/i&gt;</name>
          <visibility>0</visibility><styleUrl>#noicon</styleUrl>
          <Link><href>nodes-off.kml</href></Link>
        </NetworkLink>
      </Folder>
      <Folder id="links">
        <name>Link color...</name>
        <visibility>0</visibility><styleUrl>#radio</styleUrl>""")
        if self.LB0 is not None:
            f.write("""
        <NetworkLink>
          <name>&lt;i&gt;by network distance and betweenness&lt;/i&gt;</name>
          <visibility>0</visibility><styleUrl>#noicon</styleUrl>
          <Link><href>links-betweenness.kml</href></Link>
        </NetworkLink>
        <NetworkLink>
          <name>&lt;i&gt;yellow&lt;/i&gt;</name>
          <visibility>0</visibility><styleUrl>#noicon</styleUrl>""")
        else:
            f.write("""
        <NetworkLink>
          <name>&lt;i&gt;yellow&lt;/i&gt;</name>
          <visibility>0</visibility><styleUrl>#noicon</styleUrl>""")
        f.write("""
          <Link><href>links-yellow.kml</href></Link>
        </NetworkLink>
        <NetworkLink>
          <name>&lt;i&gt;dark blue&lt;/i&gt;</name>
          <visibility>0</visibility><styleUrl>#noicon</styleUrl>
          <Link><href>links-black.kml</href></Link>
        </NetworkLink>
      </Folder>
      <Folder id="labels">
        <name>Labels...</name>
        <visibility>0</visibility><styleUrl>#radio</styleUrl>
        <NetworkLink>
          <name>&lt;i&gt;large but simple&lt;/i&gt;</name>
          <visibility>0</visibility><styleUrl>#noicon</styleUrl>
          <Link><href>labels-large.kml</href></Link>
        </NetworkLink>
        <NetworkLink>
          <name>&lt;i&gt;small with statistics&lt;/i&gt;</name>
          <visibility>0</visibility><styleUrl>#noicon</styleUrl>
          <Link><href>labels-small.kml</href></Link>
        </NetworkLink>
        <NetworkLink>
          <name>&lt;i&gt;off&lt;/i&gt;</name>
          <visibility>0</visibility><styleUrl>#noicon</styleUrl>
          <Link><href>labels-off.kml</href></Link>
        </NetworkLink>
      </Folder>
    </Folder>
    <NetworkLink>
      <name>&lt;small&gt;(don't deselect this)&lt;/small&gt;</name>
      <open>0</open><visibility>1</visibility><refreshVisibility>0</refreshVisibility><styleUrl>#hide</styleUrl>
      <Link><href>data.kmz</href></Link>
    </NetworkLink>
  </Document>
</kml>""")
        f.close()
        f = file(path+"/regions-off.kml", "w")
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<NetworkLinkControl>
  <Update>
    <targetHref>data.kmz</targetHref>
    <Change>
      <Style targetId="a"><PolyStyle><outline>0</outline></PolyStyle></Style>
    </Change>
  </Update>
</NetworkLinkControl>
</kml>""")
        f.close()
        f = file(path+"/links-off.kml", "w")
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<NetworkLinkControl>
  <Update>
    <targetHref>data.kmz</targetHref>
    <Change>
      <Folder targetId="links"><visibility>0</visibility></Folder>
    </Change>
  </Update>
</NetworkLinkControl>
</kml>""")
        f.close()

        # determine used regions and links:
        self.regions = self.solution_nodes[0]+[]
        pa = self.solution_nodes[0]+[]
        while len(pa) < np.max(self.solutions):
            c = max(pa)
            pa.remove(c)
            C = self.children[c]
            pa += C
            self.regions += C
        links = []
        for i in self.regions:
            for j in self.regions:
                if j < i < self.parent[j] and self.linked_weight[i, j] > 0:
                    links.append((i, j))

        if self.B0 is not None:
            bmax = np.log(0.0001+max(self.B[self.regions]))
            bmin = np.log(0.0001+min(self.B[self.regions]))
            ccmax = max(self.CC[self.regions])
            f = file(path+"/nodes-betweenness.kml", "w")
            f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<NetworkLinkControl>
  <Update>
    <targetHref>data.kmz</targetHref>
    <Change>""")
            for i in self.regions:
                b = int((np.log(0.0001+self.B[i])-bmin)/(bmax-bmin)*512)
                if b < 256:
                    c = hex(511-b)[-2:] + "ffff"
                else:
                    c = "00" + hex(767-b)[-2:] + "ff"
                a = hex(256+int(self.CC[i]/ccmax*255))[-2:]
                f.write("""
<IconStyle targetId="i"""+str(i)+""""><color>"""+a+c+"""</color><scale>1.5</scale></IconStyle>""")
            f.write("""
    </Change>
  </Update>
</NetworkLinkControl>
</kml>""")
            f.close()
        f = file(path+"/nodes-yellow.kml", "w")
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<NetworkLinkControl>
  <Update>
    <targetHref>data.kmz</targetHref>
    <Change>""")
        for i in self.regions:
            f.write("""
<IconStyle targetId="i"""+str(i)+""""><color>ff00ffff</color><scale>1.5</scale></IconStyle>""")
        f.write("""
    </Change>
  </Update>
</NetworkLinkControl>
</kml>""")
        f.close()
        f = file(path+"/nodes-black.kml", "w")
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<NetworkLinkControl>
  <Update>
    <targetHref>data.kmz</targetHref>
    <Change>""")
        for i in self.regions:
            f.write("""
<IconStyle targetId="i"""+str(i)+""""><color>ff000000</color><scale>1.5</scale></IconStyle>""")
        f.write("""
    </Change>
  </Update>
</NetworkLinkControl>
</kml>""")
        f.close()
        f = file(path+"/nodes-off.kml", "w")
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<NetworkLinkControl>
  <Update>
    <targetHref>data.kmz</targetHref>
    <Change>""")
        for i in self.regions:
            f.write("""
<IconStyle targetId="i"""+str(i)+""""><scale>0</scale></IconStyle>""")
        f.write("""
    </Change>
  </Update>
</NetworkLinkControl>
</kml>""")
        f.close()
        if self.LB0 is not None:
            bmin = np.inf
            bmax = -np.inf
            for i, j in links:
                bmax = max(bmax, self.LB[i, j])
                bmin = min(bmin, self.LB[i, j])
            bmax = np.log(0.0001+bmax)
            bmin = np.log(0.0001+bmin)
            if self.D0 is not None:
                dmax = 0.0
                for i, j in links:
                    dmax = max(dmax, self.D[i, j])
                print("DMAX", dmax)
            f = file(path+"/links-betweenness.kml", "w")
            f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<NetworkLinkControl>
  <Update>
    <targetHref>data.kmz</targetHref>
    <Change>""")
            for i, j in links:
                b = int((np.log(0.0001+self.LB[i, j])-bmin)/(bmax-bmin)*512)
                if b < 256:
                    c = hex(511-b)[-2:] + "ffff"
                else:
                    c = "00" + hex(767-b)[-2:] + "ff"
                if self.D0 is None:
                    a = "ff"
                elif np.isnan(self.D[i, j]):
                    a = "ff"
                    print(i, j, "nan")
                else:
                    a = hex(511-int((self.D[i, j]-1)/(dmax-1)*255))[-2:]
                f.write("""
<LineStyle targetId="a"""+str(i)+"-"+str(j)+""""><color>"""+a+c+"""</color></LineStyle>""")
            f.write("""
    </Change>
  </Update>
</NetworkLinkControl>
</kml>""")
            f.close()
        f = file(path+"/links-yellow.kml", "w")
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<NetworkLinkControl>
  <Update>
    <targetHref>data.kmz</targetHref>
    <Change>""")
        for i, j in links:
            f.write("""
<LineStyle targetId="a"""+str(i)+"-"+str(j)+""""><color>ff00ffff</color></LineStyle>""")
        f.write("""
    </Change>
  </Update>
</NetworkLinkControl>
</kml>""")
        f.close()
        f = file(path+"/links-black.kml", "w")
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<NetworkLinkControl>
  <Update>
    <targetHref>data.kmz</targetHref>
    <Change>""")
        for i, j in links:
            f.write("""
<LineStyle targetId="a"""+str(i)+"-"+str(j)+""""><color>ff7f0000</color></LineStyle>""")
        f.write("""
    </Change>
  </Update>
</NetworkLinkControl>
</kml>""")
        f.close()
        f = file(path+"/labels-large.kml", "w")
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<NetworkLinkControl>
  <Update>
    <targetHref>data.kmz</targetHref>
    <Change>""")
        for i in self.regions:
            f.write("""
<Placemark targetId="p"""+str(i)+""""><name>"""+self.label[i]+"""</name></Placemark><LabelStyle targetId="l"""+str(i)+""""><scale>3</scale></LabelStyle>""")
        for i, j in links:
            f.write("""
<Placemark targetId="p"""+str(i)+"-"+str(j)+""""><name>"""+self.label[i]+" - "+self.label[j]+"""</name></Placemark><LabelStyle targetId="l"""+str(i)+"-"+str(j)+""""><scale>2</scale></LabelStyle>""")
        f.write("""
    </Change>
  </Update>
</NetworkLinkControl>
</kml>""")
        f.close()
        f = file(path+"/labels-small.kml", "w")
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<NetworkLinkControl>
  <Update>
    <targetHref>data.kmz</targetHref>
    <Change>""")
        for i in self.regions:
            l = self.label[i] + \
                (", dg %(k)6.2f, cls %(C)2d, btw %(B)5.2f" % {
                    "k": self.k[i]*510.07,
                    "C": self.C[i]*100,
                    "B": self.B[i]*100}).\
                replace("  ", " ").replace("  ", " ")
            # TODO: use provided data
            f.write("""
<Placemark targetId="p"""+str(i)+""""><name>"""+l+"""</name></Placemark><LabelStyle targetId="l"""+str(i)+""""><scale>3</scale></LabelStyle>""")
        for i, j in links:
            l = self.label[i] + " - " + self.label[j] + \
                (", str %(s)5.2f, dst %(l)6.2f, btw %(B)7.4f" % {
                    "s": self.linked_weight[i, j]*510.07**2,
                    "l": self.D[i, j],
                    "B": self.LB[i, j]*100}).\
                replace("  ", " ").replace("  ", " ")
            # TODO: use provided data
            f.write("""
<Placemark targetId="p"""+str(i)+"-"+str(j)+""""><name>"""+l+"""</name></Placemark><LabelStyle targetId="l"""+str(i)+"-"+str(j)+""""><scale>2</scale></LabelStyle>""")
        f.write("""
    </Change>
  </Update>
</NetworkLinkControl>
</kml>""")
        f.close()
        f = file(path+"/labels-off.kml", "w")
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<NetworkLinkControl>
  <Update>
    <targetHref>data.kmz</targetHref>
    <Change>""")
        for i in self.regions:
            f.write("""
<Placemark targetId="p"""+str(i)+""""><name>"""+self.label[i]+"""</name></Placemark><LabelStyle targetId="l"""+str(i)+""""><scale>0</scale></LabelStyle>""")
        for i, j in links:
            f.write("""
<Placemark targetId="p"""+str(i)+"-"+str(j)+""""><name>"""+self.label[i]+" - "+self.label[j]+"""</name></Placemark><LabelStyle targetId="l"""+str(i)+"-"+str(j)+""""><scale>0</scale></LabelStyle>""")
        f.write("""
    </Change>
  </Update>
</NetworkLinkControl>
</kml>""")
        f.close()
        f = file(path+"/hierarchy-regions.kml", "w")
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<NetworkLinkControl>
  <Update>
    <targetHref>data.kmz</targetHref>
    <Change>
      <Style targetId="a"><PolyStyle><outline>1</outline></PolyStyle></Style>""")
        for i in self.regions:
            t1 = max(np.min(self.solutions), self.S+1-self.parent[i])
            t2 = min(self.S+1-i, np.max(self.solutions)+1)-1
            f.write("""
<Placemark targetId="p"""+str(i)+""""><TimeSpan><begin>"""+str(t1)+"""</begin><end>"""+str(t2)+"""-12-31</end></TimeSpan><LookAt><gx:TimeStamp><when>"""+str(t1)+"""</when></gx:TimeStamp></LookAt></Placemark>""")
        f.write("""
      <Folder targetId="regions"><visibility>1</visibility></Folder>
    </Change>
  </Update>
</NetworkLinkControl>
</kml>""")
        f.close()
        f = file(path+"/hierarchy-links.kml", "w")
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<NetworkLinkControl>
  <Update>
    <targetHref>data.kmz</targetHref>
    <Change>""")
        for i, j in links:
            t1 = max(np.min(self.solutions), self.S+1-self.parent[i],
                     self.S+1-self.parent[j])
            t2 = min(self.S+1-i, self.S+1-j, np.max(self.solutions)+1)-1
            f.write("""
<Placemark targetId="p"""+str(i)+"-"+str(j)+""""><TimeSpan><begin>"""+str(t1)+"""</begin><end>"""+str(t2)+"""-12-31</end></TimeSpan></Placemark>""")
        f.write("""
      <Folder targetId="links"><visibility>1</visibility></Folder>
    </Change>
  </Update>
</NetworkLinkControl>
</kml>""")
        f.close()
        for level in range(len(self.solutions)):
            l = self.solutions[level]
            f = file(path+"/partition"+str(l)+"-regions.kml", "w")
            f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<NetworkLinkControl>
  <Update>
    <targetHref>data.kmz</targetHref>
    <Change>
      <Style targetId="a"><PolyStyle><outline>1</outline></PolyStyle></Style>""")
            for i in self.solution_nodes[level]:
                f.write("""
<Placemark targetId="p"""+str(i)+""""><TimeSpan><begin/><end/></TimeSpan><LookAt><gx:TimeStamp><when/></gx:TimeStamp></LookAt></Placemark>""")
            f.write("""
      <Folder targetId="regions"><visibility>1</visibility></Folder>
      <Folder targetId="regions"""+str(l)+""""><visibility>1</visibility></Folder>
      <Folder targetId="regions0"><visibility>0</visibility></Folder>""")
            for level2 in range(len(self.solutions)):
                if level2 != level:
                    f.write("""
      <Folder targetId="regions"""+str(self.solutions[level2])+""""><visibility>0</visibility></Folder>""")
            f.write("""
    </Change>
  </Update>
</NetworkLinkControl>
</kml>""")
            f.close()
            f = file(path+"/partition"+str(l)+"-links.kml", "w")
            f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<NetworkLinkControl>
  <Update>
    <targetHref>data.kmz</targetHref>
    <Change>""")
            for i in self.solution_nodes[level]:
                for j in self.solution_nodes[level]:
                    if (i, j) in links:
                        f.write("""
<Placemark targetId="p"""+str(i)+"-"+str(j)+""""><TimeSpan><begin/><end/></TimeSpan></Placemark>""")
            f.write("""
      <Folder targetId="links"><visibility>1</visibility></Folder>
      <Folder targetId="links"""+str(l)+""""><visibility>1</visibility></Folder>
      <Folder targetId="links0"><visibility>0</visibility></Folder>""")
            for level2 in range(len(self.solutions)):
                if level2 != level:
                    f.write("""
      <Folder targetId="links"""+str(self.solutions[level2])+""""><visibility>0</visibility></Folder>""")
            f.write("""
    </Change>
  </Update>
</NetworkLinkControl>
</kml>""")
            f.close()
        f = file(path+"/data.kml", "w")
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">
 <Document>
  <Folder><visibility>1</visibility></Folder>
  <Style id="a"><!-- common to regions -->
    <PolyStyle>
      <fill>0</fill><outline>1</outline>
    </PolyStyle>
    <BalloonStyle>
      <text><![CDATA[
        <b>$[l]</b><br/>
        <table>
          <tr><td>Area</td><td>$[a]&nbsp;Mm²</td></tr>
          <tr><td>Ave. degree measure</td><td>$[k]&nbsp;Mm²</td></tr>
          <tr><td>Ave. clustering measure</td><td nowrap>$[C] %</td></tr>
          <tr><td>Ave. centrality measure</td><td nowrap>$[CC] %</td></tr>
          <tr><td>Ave. betweenness measure</td><td nowrap>$[B] %</td></tr>
          <tr><td>contained in region</td><td><a href="#p$[p];balloonFlyto">$[pl]</a></td></tr>
          <tr><td>largest contained subregions</td><td><a href="#p$[n1];balloonFlyto">$[l1]</a>, <a href="#p$[n2];balloonFlyto">$[l2]</a></td></tr>
        </table>
      ]]></text>
    </BalloonStyle>
  </Style>
  <Style id="b"><!-- highlit regions -->
    <PolyStyle>
      <fill>0</fill><outline>1</outline>
    </PolyStyle>
    <LineStyle>
      <color>ff0000ff</color><width>12</width>
    </LineStyle>
    <LabelStyle>
      <color>ff1f1fff</color><scale>3.5</scale>
    </LabelStyle>
    <IconStyle>
      <color>ff3f3fff</color><scale>2.0</scale><Icon><href>i.png</href></Icon>
    </IconStyle>
    <BalloonStyle>
      <text><![CDATA[
        <b>$[l]</b><br/>
        <table>
          <tr><td>Area</td><td>$[a]&nbsp;Mm²</td></tr>
          <tr><td>Ave. degree measure</td><td>$[k]&nbsp;Mm²</td></tr>
          <tr><td>Ave. clustering measure</td><td nowrap>$[C] %</td></tr>
          <tr><td>Ave. centrality measure</td><td nowrap>$[CC] %</td></tr>
          <tr><td>Ave. betweenness measure</td><td nowrap>$[B] %</td></tr>
          <tr><td>contained in region</td><td><a href="#p$[p];balloonFlyto">$[pl]</a></td></tr>
          <tr><td>largest contained subregions</td><td><a href="#p$[n1];balloonFlyto">$[l1]</a>, <a href="#p$[n2];balloonFlyto">$[l2]</a></td></tr>
        </table>
      ]]></text>
    </BalloonStyle>
  </Style>
  <Style id="e"><!-- arcs -->
    <IconStyle>
      <color>00000000</color>
    </IconStyle>
    <BalloonStyle>
      <text><![CDATA[
        <b><a href="#p$[n1];balloonFlyto">$[l1]</a> - <a href="#p$[n2];balloonFlyto">$[l2]</a></b><br/>
        <table>
          <tr><td>Strength (total product of linked area)</td><td nowrap>$[w] (Mm²)²</td></tr>
          <tr><td>Relative strength</td><td nowrap>$[r] %</td></tr>
          <tr><td>Ave. network distance</td><td nowrap>$[l] steps</td></tr>
          <tr><td>Ave. link betweenness measure</td><td nowrap>$[B] %</td></tr>
        </table>
      ]]></text>
    </BalloonStyle>
  </Style>
  <Style id="f"><!-- highlit arcs -->
    <LineStyle>
      <color>ff0000ff</color><width>4</width>
    </LineStyle>
    <LabelStyle>
      <color>ff3f3fff</color><scale>2.5</scale>
    </LabelStyle>
    <IconStyle>
      <color>00000000</color>
    </IconStyle>
    <BalloonStyle>
      <text><![CDATA[
        <b><a href="#p$[n1];balloonFlyto">$[l1]</a> - <a href="#p$[n2];balloonFlyto">$[l2]</a></b><br/>
        <table>
          <tr><td>Strength (total product of linked area)</td><td nowrap>$[w] (Mm²)²</td></tr>
          <tr><td>Relative strength</td><td nowrap>$[r] %</td></tr>
          <tr><td>Ave. network distance</td><td nowrap>$[l] steps</td></tr>
          <tr><td>Ave. link betweenness measure</td><td nowrap>$[B] %</td></tr>
        </table>
      ]]></text>
    </BalloonStyle>
  </Style>
  <Folder id="regions">
    <visibility>0</visibility>""")
        intermediate = set(self.regions)
        intermediate_links = set(links)
        for level in range(len(self.solutions)):
            l = self.solutions[level]
            f.write("""
    <Folder id="regions"""+str(l)+"""">""")
            sn = set(self.solution_nodes[level])
            for i in sn.intersection(intermediate):
                self._write_kml_region(f, i)
            intermediate -= sn
            f.write("""
    </Folder>""")
        f.write("""
    <Folder id="regions0">""")
        for i in intermediate:
            self._write_kml_region(f, i)
        f.write("""
    </Folder>
  </Folder>
  <Folder id="links">
    <visibility>0</visibility>""")
        limax = -np.inf
        for i, j in links:
            limax = max(limax, self.linked_weight[i, j])
        for level in range(len(self.solutions)):
            l = self.solutions[level]
            f.write("""
    <Folder id="links"""+str(l)+"""">""")
            for i in self.solution_nodes[level]:
                for j in self.solution_nodes[level]:
                    if (i, j) in intermediate_links:
                        intermediate_links.remove((i, j))
                        self._write_kml_link(f, i, j,
                                             self.linked_weight[i, j]/limax)
            f.write("""
    </Folder>""")
        f.write("""
    <Folder id="links0">""")
        for i, j in intermediate_links:
            self._write_kml_link(f, i, j, self.linked_weight[i, j]/limax)
        f.write("""
    </Folder>
  </Folder>
 </Document>
</kml>""")
        f.close()
        os.system("zip -9m "+path+"/data.kmz "+path+"/data.kml")

    def _write_kml_region(self, f, i):
        if self.parent[i] in self.regions:
            p = str(self.parent[i])
            pl = self.label[self.parent[i]]
        else:
            p = pl = ""
        if len(self.children[i]) > 1:
            c1 = self.children[i][0]
            if c1 in self.regions:
                l1 = self.label[c1]
                c1 = str(c1)
            else:
                l1 = c1 = ""
            c2 = self.children[i][1]
            if c2 in self.regions:
                l2 = self.label[c2]
                c2 = str(c2)
            else:
                l2 = c2 = ""
        else:
            l1 = c1 = l2 = c2 = ""
        f.write("""
<Placemark id="p"""+str(i)+""""><name/><ExtendedData><Data name="l"><value>
"""+self.label[i]+"""
</value></Data><Data name="a"><value>
"""+str(round(self.weight[i]*51007)/100)+"""
</value></Data><Data name="k"><value>
"""+str(round(self.k[i]*51007)/100)+"""
</value></Data><Data name="C"><value>
"""+str(int(100*self.C[i]))+"""
</value></Data><Data name="CC"><value>
"""+str(int(100*self.CC[i]))+"""
</value></Data><Data name="B"><value>
"""+str(round(10000*self.B[i])/100)+"""
</value></Data><Data name="p"><value>
"""+p+"""
</value></Data><Data name="pl"><value>
"""+pl+"""
</value></Data><Data name="n1"><value>
"""+c1+"""
</value></Data><Data name="l1"><value>
"""+l1+"""
</value></Data><Data name="n2"><value>
"""+c2+"""
</value></Data><Data name="l2"><value>
"""+l2+"""
</value></Data></ExtendedData><TimeSpan/><LookAt><gx:TimeStamp><when/></gx:TimeStamp><longitude>
"""+str(self.lon[i])+"""
</longitude><latitude>
"""+str(self.lat[i])+"""
</latitude><range>12000000</range></LookAt><styleUrl>#a</styleUrl><StyleMap><Pair><key>normal</key><Style><IconStyle id="i"""+str(i)+""""><color/>
<scale>1</scale><Icon><href>i.png</href></Icon></IconStyle><LabelStyle id="l"""+str(i)+""""><color>
ff"""+self.color[i][1:7]+"""
</color><scale>0</scale></LabelStyle><LineStyle><color>
ff"""+self.color[i][1:7]+"""
</color><width>6</width></LineStyle></Style></Pair><Pair><key>highlight</key><styleUrl>#b</styleUrl></Pair></StyleMap><MultiGeometry><Point><coordinates>
"""+str(self.lon[i])+","+str(self.lat[i])+"""
</coordinates></Point>""")
        if self.shape[i] is not None:
            for sh in self.shape[i]:
                f.write("""
<Polygon><tessellate>1</tessellate><outerBoundaryIs><LinearRing><coordinates>""")
                for la, lo in sh:
                    f.write("\n"+str(round(100*lo)/100) +
                            "," + str(round(100*la)/100))
                f.write("""
</coordinates></LinearRing></outerBoundaryIs></Polygon>""")
        f.write("""
</MultiGeometry></Placemark>""")

    def _write_kml_link(self, f, i, j, relwidth=0.1):
        nps = 21
        posi = np.array(
            [self.network.latlon2cartesian(self.lat[i], self.lon[i])])
        posj = np.array(
            [self.network.latlon2cartesian(self.lat[j], self.lon[j])])
        pos = posi + (posj-posi) * np.linspace(0, 1, nps).reshape((-1, 1))
        hgt = (.25 - (.5-np.linspace(0, 1, nps))**2) * 10000000 * \
            (0.05+((posi-posj)**2).sum()/4*0.95)
        pos /= np.sqrt((pos**2).sum(axis=1)).reshape((-1, 1))
        la, lo = self.network.cartesian2latlon(pos[nps/2, :])
        hotpos = str(round(100*lo)/100) + "," + str(round(100*la)/100) + \
            "," + str(round(hgt[nps/2]))
        d = self.D[i, j]
        if d > 0:
            d = str(round(d*100)/100)
        else:
            d = ""
        lb = self.LB[i, j]
        if lb > 0:
            lb = str(round(1000000*lb)/10000)
        else:
            lb = ""
        f.write("""
<Placemark id="p"""+str(i)+"-"+str(j)+""""><name/><ExtendedData><Data name="w"><value>
"""+str(round(self.linked_weight[i, j]*510.07**2*100)/100)+"""
</value></Data><Data name="r"><value>
"""+str(round(self.linked_proportion[i, j]*10000)/100)+"""
</value></Data><Data name="l"><value>
"""+d+"""
</value></Data><Data name="B"><value>
"""+lb+"""
</value></Data><Data name="n1"><value>
"""+str(i)+"""
</value></Data><Data name="l1"><value>
"""+self.label[i]+"""
</value></Data><Data name="n2"><value>
"""+str(i)+"""
</value></Data><Data name="l2"><value>
"""+self.label[j]+"""
</value></Data></ExtendedData><TimeSpan/><styleUrl>#e</styleUrl><StyleMap><Pair><key>normal</key><Style><LabelStyle id="l"""+str(i)+"-"+str(j)+""""><color>bf7fffff</color><scale>0</scale></LabelStyle><LineStyle id="a"""+str(i)+"-"+str(j)+""""><color/><width>
"""+str(relwidth*5+1)+"""
</width></LineStyle></Style></Pair><Pair><key>highlight</key><styleUrl>#f</styleUrl></Pair></StyleMap><MultiGeometry><Point><altitudeMode>relativeToGround</altitudeMode><coordinates>
"""+hotpos+"""
</coordinates></Point><LineString><altitudeMode>relativeToGround</altitudeMode><coordinates>""")
        for s in range(nps):
            la, lo = self.network.cartesian2latlon(pos[s, :])
            f.write("\n" + str(round(100*lo)/100) +
                    "," + str(round(100*la)/100) + "," + str(int(hgt[s])))
        f.write("""
</coordinates></LineString></MultiGeometry></Placemark>""")

    def _improve_step(self):
        """improve layout by one step"""
        # TODO: fruchterman or some such
        for i in self.shown:
            posi = self.position[i]
            for j, line in self.lines[i].items():
                if j > i:
                    posj = posi
                    # disterr = (((posi-posj)**2).sum()-1000000.0)**2
        self._update()

    def _process_expand_button_on(self, i):
        """expand a shown supernode and show collapsed descendants"""
        if i is None or i < self.N:
            return
        self._expand(i)
        self._hide(i)
        S = set([i])
        while len(S) > 0:
            for c in self.children[S.pop()]:
                if self.is_expanded[c]:
                    S.add(c)
                else:
                    # TODO: compute x,y more sensibly
                    if self.position[c, 0] == 0.0 and not self.fixed:
                        r = 4*self.radius[i] - self.radius[c]
                        phi = 2*np.pi * random.uniform()  # c/self.S
                        pos = self.position[i, :] + \
                            r * np.array([np.cos(phi), np.sin(phi)])
                        self._show(c, pos)
                    else:
                        self._show(c, self.position[c])

    def _process_collapse_button_on(self, i):
        """collapse parent of a shown supernode and hide descendants"""
        if i is None or i == self.S-1:
            return
        p = self.parent[i]
        self._collapse(p)
        S = set([p])
        sumweights = 0.0
        sumpos = np.zeros((self.dim))
        while len(S) > 0:
            for c in self.children[S.pop()]:
                if self.is_expanded[c]:
                    S.add(c)
                else:
                    sumweights += self.weight[c]
                    sumpos += self.weight[c] * self.position[c]
                    self._hide(c)
        if self.position[p, 0] == 0.0 and not self.fixed:
            self._show(p, sumpos/sumweights)
        else:
            self._show(p, self.position[p])

    def _show(self, i, pos):
        """
        show a supernode at some position and incident edges to other shown
        supernodes
        """
        if self.is_shown[i]:
            return False
        self.is_shown[i] = True
        self.shown.add(i)
        self.position[i] = pos
        color = self.color[i]
        self.circles[i] = mpl.pyplot.Circle(
            (pos[0], pos[1]), radius=self.radius[i], color=color, alpha=0.5)
        self.ax.add_patch(self.circles[i])
        if self.map:
            self.polys[i] = []
            for sh in self.shape[i]:
                sh = np.array(sh)
                x, y = self.bm(sh[:, 1], sh[:, 0])
                dists2 = (x[1:]-x[:-1])**2 + (y[1:]-y[:-1])**2
                jumps = np.where(dists2 > 50*dists2.min())[0]
                if len(jumps) > 1:
                    self.polys[i] += \
                        self.bm.plot(list(sh[1+jumps[-1]:, 1]) +
                                     list(sh[:1+jumps[0], 1]),
                                     list(sh[1+jumps[-1]:, 0]) +
                                     list(sh[:1+jumps[0], 0]),
                                     color=color, alpha=0.25, lw=3.0)
                    for j in range(len(jumps)-1):
                        self.polys[i] += \
                            self.bm.plot(sh[1+jumps[j]:1+jumps[j+1], 1],
                                         sh[1+jumps[j]:1+jumps[j+1], 0],
                                         color=color, alpha=0.25, lw=3.0)
                else:
                    self.polys[i] += \
                        self.bm.plot(sh[:, 1], sh[:, 0],
                                     color=color, alpha=0.25, lw=3.0)
        for j in self.shown:
            if j == i:
                continue
            lw = self.linked_weight[i, j]
            if lw > 0.0:
                color = "gray"
                self.lines[i][j] = self.lines[j][i] = line \
                    = mpl.pyplot.Line2D(self.position[[i, j], 0],
                                        self.position[[i, j], 1], alpha=0.5,
                                        color=color, linewidth=0.5+lw*99)
                self.ax.add_artist(line)
        self.fig.show()
        return True

    def _hide(self, i):
        """hide a supernode and incident edges"""
        if not self.is_shown[i]:
            return False
        self.is_shown[i] = False
        self.shown.remove(i)
        self.circles[i].remove()
        self.circles[i] = None
        if self.is_geo:
            for p in self.polys[i]:
                p.remove()
                del p
            self.polys[i] = []
        for j, line in self.lines[i].items():
            line.remove()
            del self.lines[j][i]
        self.lines[i] = {}
        self.fig.show()
        return True

    def _move(self, i, pos):
        """move a shown supernode to some position and adjust incident edges"""
        self.position[i] = pos
        self.circles[i].center = (pos[0], pos[1])
        for j, line in self.lines[i].items():
            line.set_data(self.position[[i, j], :].T)
        self.fig.show()

    def _update(self):
        for i in self.shown:
            pos = self.position[i]
            self.circles[i].center = (pos[0], pos[1])
            for j, line in self.lines[i].items():
                if j > i:
                    line.set_data(self.position[[i, j], :].T)
        self.fig.show()

    def _expand(self, i):
        """expand a supernode and return whether its state changed"""
        if self.is_expanded[i]:
            return False
        self.is_expanded[i] = True
        return True

    def _collapse(self, i):
        """collapse a supernode and return whether its state changed"""
        if not self.is_expanded[i]:
            return False
        self.is_expanded[i] = False
        return True

    def _dummy():
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
 <Document>
  <Style id="a"><!-- common to regions -->
   <PolyStyle>
    <fill>0</fill>
    <outline>1</outline>
   </PolyStyle>
   <IconStyle>
    <color>00000000</color>
   </IconStyle>
   <BalloonStyle>
    <text><![CDATA[
     <b>$[name]</b><br/>
     <table>
      <tr><td>Area</td><td><b>a</b></td><td nowrap>$[a] Mm²</td></tr>
      <tr><td>Ave. linked area (n.s.i. degree)</td><td><b>k</b></td><td nowrap>$[k] Mm²</td></tr>
      <tr><td>Ave. n.s.i. clustering coeff.</td><td><b>C</b></td><td nowrap>$[C] %</td></tr>
      <tr><td>Ave. n.s.i. shortest path betweenness</td><td><b>B</b></td><td nowrap>$[B] %</td></tr>
      <tr><td>contained in region</td><td/><td>$[p]</td></tr>
      <tr><td>used in solution(s) no.</td><td/><td>$[t]</td></tr>
      <tr><td>contains regions</td><td/><td>$[ch]</td></tr>
     </table>
    ]]></text>
   </BalloonStyle>
  </Style>
  <Style id="b"><!-- highlit regions -->
   <PolyStyle>
    <fill>0</fill>
    <outline>1</outline>
   </PolyStyle>
   <LineStyle>
    <color>ff0000ff</color>
    <width>12</width>
   </LineStyle>
   <LabelStyle>
    <color>ff3f3fff</color>
    <scale>4.0</scale>
   </LabelStyle>
   <IconStyle>
    <color>00000000</color>
   </IconStyle>
   <BalloonStyle>
    <text><![CDATA[
     <b>$[name]</b><br/>
     <table>
      <tr><td>Area</td><td><b>a</b></td><td>$[a]&nbsp;Mm²</td></tr>
      <tr><td>Ave. linked area (n.s.i. degree)</td><td><b>k</b></td><td>$[k]&nbsp;Mm²</td></tr>
      <tr><td>Ave. n.s.i. clustering coeff.</td><td><b>C</b></td><td nowrap>$[C] %</td></tr>
      <tr><td>Ave. n.s.i. shortest path betweenness</td><td><b>B</b></td><td nowrap>$[B] %</td></tr>
      <tr><td>contained in region</td><td/><td>$[p]</td></tr>
      <tr><td>used in solution(s) no.</td><td/><td>$[t]</td></tr>
      <tr><td>contains regions</td><td/><td>$[ch]</td></tr>
     </table>
    ]]></text>
   </BalloonStyle>
  </Style>
  <Style id="c"><!-- common to nodes -->
   <IconStyle>
    <color>00000000</color>
   </IconStyle>
  </Style>
  <Style id="d"><!-- highlit nodes -->
   <LabelStyle>
    <color>ff3f3fff</color>
    <scale>4.0</scale>
   </LabelStyle>
   <IconStyle>
    <color>00000000</color>
   </IconStyle>
  </Style>
  <Style id="e"><!-- arcs -->
   <LabelStyle>
    <color>bf7fffff</color>
    <scale>1.0</scale>
   </LabelStyle>
   <IconStyle>
    <color>00000000</color>
   </IconStyle>
   <BalloonStyle/>
    <!--<text><![CDATA[
      <b>$[name]</b><br/>
      <table>
      <tr><td>Strength</td><td><b>a</b></td><td nowrap>$[s] (Mm²)²</td></tr>
      <tr><td>Relative strength</td><td><b>a</b></td><td nowrap>$[r] %</td></tr>
      <tr><td>Ave. network distance</td><td><b>l</b></td><td nowrap>$[l] steps</td></tr>
      <tr><td>Ave. n.s.i. shortest path betweenness</td><td><b>B</b></td><td nowrap>$[B] %</td></tr>
      <tr><td>used in solution(s) no.</td><td/><td>$[t]</td></tr>
      </table>
    ]]></text>
   </BalloonStyle>-->
  </Style>
  <Style id="f"><!-- highlit arcs -->
   <LineStyle>
    <color>ff0000ff</color>
    <width>4</width>
   </LineStyle>
   <LabelStyle>
    <color>ff3f3fff</color>
    <scale>1.5</scale>
   </LabelStyle>
   <IconStyle>
    <color>00000000</color>
    </IconStyle>
   <BalloonStyle/>
    <!--<text><![CDATA[
      <b>$[name]</b><br/>
      <table>
      <tr><td>Strength</td><td><b>a</b></td><td nowrap>$[s] (Mm²)²</td></tr>
      <tr><td>Relative strength</td><td><b>a</b></td><td nowrap>$[r] %</td></tr>
      <tr><td>Ave. network distance</td><td><b>l</b></td><td nowrap>$[l] steps</td></tr>
      <tr><td>Ave. n.s.i. shortest path betweenness</td><td><b>B</b></td><td nowrap>$[B] %</td></tr>
      <tr><td>used in solution(s) no.</td><td/><td>$[t]</td></tr>
      </table>
    ]]></text>
   </BalloonStyle>-->
  </Style>
  <Style id="h"><!-- hidden children -->
   <ListStyle>
    <listItemType>checkHideChildren</listItemType>
   </ListStyle>
  </Style>
  <Style id="l">
   <ListStyle>
    <ItemIcon/>
   </ListStyle>
  </Style>
  <Style id="lh">
   <ListStyle>
    <listItemType>checkHideChildren</listItemType>
   </ListStyle>
  </Style>
  <Style id="lr">
   <ListStyle>
    <listItemType>radioFolder</listItemType>
   </ListStyle>
  </Style>
  <Folder id="0">
   <name>"""+title+"""</name>
   <styleUrl>#lr</styleUrl>
   <Folder id="h">
    <name>Slideable Hierarchy of Regions</name>
    <description><![CDATA[
     region labels inside subfolders give this information:<br/>
     <b>round
      <a href="#h;balloon">code</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      <a href="#h;balloon">a</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      <a href="#h;balloon">k</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      <a href="#h;balloon">C</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      <a href="#h;balloon">B</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      <a href="#h;balloon">location</a>
     </b><br/><br/>
     <table>
      <tr>
       <td nowrap><i>
        code<br/>
        a<br/>
        k<br/>
        C<br/>
        B<br/>
        location
       </i></td><td nowrap>
        is a short unique code,<br/>
        is the region's area,<br/>
        is the region's average linked area (n.s.i. degree),<br/>
        is the region's average n.s.i. clustering coeff.,<br/>
        is the region's average n.s.i. shortest path betweenness,<br/>
        gives its approximate location.
       </td>
      </tr>
     </table>
     <h1>Slideable Hierarchy of Regions</h1>
     <p>This folder contains a hierarchy of regions covering the Earth's surface.
     Use the slider on the top-left to successively increase the level of detail
     from a coarse partition into """+str(len(self.solution_nodes[0]))+""" regions
     to a fine partition into """+str(len(self.solution_nodes[-1]))+""" regions
     (if the slider shows a "month" and "day", just ignore them).
     <p>Expand this folder to get more information on individual regions,
     ordered into subfolders corresponding to certain selected levels of detail.
     The hierarchy can be used in combination with third-party overlays that do not themselves use the slider.
     (For third-party overlays using the slider, use the folder <a href="#p;balloonFlyto">Selected Partitions</a> instead.)
    ]]></description>
    <visibility>1</visibility>""")
        f.write("""
    </Folder>
   </Folder>
   <Folder id="p">
    <name>Selected Partitions</name>
    <description><![CDATA[
     region labels inside subfolders give this information:<br/>
     <b>
      <a href="#p;balloon">code</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      <a href="#p;balloon">a</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      <a href="#p;balloon">k</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      <a href="#p;balloon">C</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      <a href="#p;balloon">B</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      <a href="#p;balloon">location</a>
     </b><br/><br/>
     <table>
      <tr>
       <td nowrap><i>
        code<br/>
        a<br/>
        k<br/>
        C<br/>
        B<br/>
        location
       </i></td><td nowrap>
        is a short unique code,<br/>
        is the region's area,<br/>
        is the region's average linked area (n.s.i. degree),<br/>
        is the region's average n.s.i. clustering coeff.,<br/>
        is the region's average n.s.i. shortest path betweenness,<br/>
        gives its approximate location.
       </td>
      </tr>
     </table>
     <h1>Selected Partitions</h1>
     <p>This folder contains sets of regions that each cover the Earth's surface
     and correspond to certain selected levels of detail from the above hierarchy.
     They can be used in combination with third-party overlays that use a slider.
     Expand the subfolders to get more information on individual regions.
     <a href="r5063;balloonFlyto">TEST</a>
    ]]></description>
    <styleUrl>#lr</styleUrl>""")
