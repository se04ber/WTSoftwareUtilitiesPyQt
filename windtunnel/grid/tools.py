#! /usr/bin/python3
# -*- coding: utf-8 -*-
""" 
/grid/tools.py: 
generates measurement grids for domains with obstacles/buildings.
"""
from __future__ import print_function, division
from itertools import islice
from array import array as pyarray
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv

__all__ = [
    'building',
    'configuration',
    'intersects',
    'get_metangle',
    'cost_func',
    'optimize_grid',
    'rotate_origin_only',
    'rotate_via_numpy'
]


class building():
    """
    Class used to describe the properties of a building. Used to easier plot the different buildings in a flow or
    concentration experiment.
    """

    def __init__(self, name, type, x, y, x_extent, y_extent, z_extent, global_transform=0):
        self.name = name
        self.type = type
        self.x_og = x
        self.y_og = y
        self.x_pos,self.y_pos = rotate_origin_only(x,y,global_transform)
        self.x_extent = x_extent
        self.y_extent = y_extent
        self.z_extent = z_extent # will be probably stay unused for a while (unless side views are needed)
        self.global_transform = global_transform
        if global_transform!=0:

            self.coords = np.zeros([4,2])
            self.coords[0][0] = x
            self.coords[0][1] = y

            self.coords[1][0] = x + x_extent
            self.coords[1][1] = y

            self.coords[2][0] = x + x_extent
            self.coords[2][1] = y + y_extent

            self.coords[3][0] = x
            self.coords[3][1] = y + y_extent

            self.coords_transformed = rotate_via_numpy(self.coords,self.global_transform)

        self.boundaries=[]
        self.calc_boundaries(global_transform)

        if type == 'rect':
            if global_transform==0:
                self.patch = patches.Rectangle((self.x_pos, self.y_pos), 
                self.x_extent, self.y_extent,edgecolor='none',linewidth=1.5, fill=1,
                facecolor=[0,0,0,0.5],alpha=0.4)
            else:
                self.patch = patches.Polygon(self.coords_transformed,
                True,edgecolor='none',linewidth=1.5, fill=1,facecolor=[0,0,0,0.5],alpha=0.4)
        elif type == 'cyld':
            self.patch = patches.Ellipse((self.x_pos,self.y_pos),self.x_extent,
            self.y_extent,angle=global_transform,edgecolor='none',linewidth=1.5, 
            fill=1,facecolor=[0,0,0,0.5],alpha=0.4)
    def calc_boundaries(self,global_transform):
        '''
        Calculates the building boundaries from the positions and extents.

        Returns
        -------
        '''
        if global_transform==0:
            self.boundaries.append([[self.x_pos,self.y_pos],
                                [self.x_pos+self.x_extent,self.y_pos]]) #lower
            self.boundaries.append([[self.x_pos+self.x_extent,self.y_pos],
                                [self.x_pos + self.x_extent, self.y_pos + self.y_extent]]) # right
            self.boundaries.append([[self.x_pos + self.x_extent, self.y_pos + self.y_extent],
                                [self.x_pos, self.y_pos + self.y_extent]])  # upper
            self.boundaries.append([[self.x_pos, self.y_pos + self.y_extent],
                                [self.x_pos, self.y_pos]])  # left
        else:
            self.boundaries.append([self.coords_transformed[0, :], self.coords_transformed[1, :]])
            self.boundaries.append([self.coords_transformed[1, :], self.coords_transformed[2, :]])
            self.boundaries.append([self.coords_transformed[2, :], self.coords_transformed[3, :]])
            self.boundaries.append([self.coords_transformed[3, :], self.coords_transformed[0, :]])

    def refresh_patches(self):
        if self.type == 'rect':
            if self.global_transform==0:
                self.patch = patches.Rectangle((self.x_pos, self.y_pos), self.x_extent, self.y_extent,
                                            edgecolor='none',linewidth=1.5, fill=1,facecolor=[0,0,0,0.5],alpha=0.4)
            else:
                self.patch = patches.Polygon(self.coords_transformed,
                                             True, edgecolor='none', linewidth=1.5, fill=1, facecolor=[0, 0, 0, 0.5],
                                             alpha=0.4)
        elif self.type == 'cyld':
            self.patch = patches.Ellipse((self.x_pos,self.y_pos),self.x_extent,self.y_extent,angle=self.global_transform,
                                         edgecolor='none',linewidth=1.5, fill=1,facecolor=[0,0,0,0.5],alpha=0.4)
class configuration():
    '''
    Takes positions, extents and types of buildings from a .csv file and generates a configuration which can be used for
    plotting and grid generation. To initialize an object of the configuration class only a .csv file of the building
    parameters is needed. The following structure is required:
    Name    type(rect)  x_pos   y_pos   x_extent    y_extent    z_extent    orientation
    '''

    def __init__(self,path,global_angle=0):

        self.buildings = []
        self.global_angle=global_angle
        with open(path,'r') as file:
            for line in csv.reader(file,
                                   delimiter="\t"):  # You can also use delimiter="\t" rather than giving a dialect.
                print(line)
                self.buildings.append(building(line[0], line[1], float(line[2]), float(line[3]), float(line[4]),
                                                  float(line[5]), float(line[6]), global_angle))

        self.domain_extents = np.array([np.min([struct.x_pos for struct in self.buildings]) - 200,
                     np.max([struct.x_pos + struct.x_extent for struct in self.buildings]) + 200,
                                      np.min([struct.y_pos for struct in self.buildings]) - 200,
                                      np.max([struct.y_pos + struct.y_extent for struct in self.buildings]) + 200])

    def plot_configuration(self,figure_size):
        '''
        Plots the buildings in the configuration with a desired figure size
        Parameters
        ----------
        figure_size

        Returns
        -------

        '''
        fig, ax = plt.subplots(1,figsize=figure_size)
        ax.grid(linewidth=0.5)

        for struct in self.buildings:
            struct.refresh_patches()
            ax.add_patch(struct.patch)

        ax.set_xlim(self.domain_extents[0:2])

        ax.set_ylim(self.domain_extents[2:])
        return fig,ax

    def add_configuration(self,ax):
        '''
        Adds the building configuration to the ax.

        Parameters
        ----------
        ax

        Returns
        -------

        '''
        ax.grid(linewidth=0.5)
        for struct in self.buildings:
            ax.add_patch(struct.patch)

        ax.set_xlim(self.domain_extents[0:2])

        ax.set_ylim(self.domain_extents[2:])
        return ax

    def gen_polar_grid(self,angles,dists,z,x_offset=0,y_offset=0,avoid_buildings=True,tolerance=0):
        '''
        Generates a polar grid of points from the input parameters. If desired the points that lie inside of
        buildings can be omitted. If the origin of the polar grid is not on (0,0) adjust the x_offset and y_offset
        parameters to shift the centre of the polar grid.

        Parameters
        ----------
        angles: array_like
            angles in degrees
        dists: array_like
            distances
        z: array_like
        x_offset: float
        y_offset: float
        avoid_buildings: bool

        Returns
        -------
        grid: array_like
                2-D array representing the coordinates of the points [X,Y,Z]
        '''
        a, d = np.meshgrid(angles, dists)

        x = np.outer(d, np.cos(np.radians(a)))-x_offset
        y = np.outer(d, np.sin(np.radians(a)))-y_offset

        if self.global_transform != 0:
            xy_transformed = rotate_via_numpy(np.stack([x, y]),self.global_transform)
            x = xy_transformed[:,0]
            y = xy_transformed[:,1]

        z,_ = np.meshgrid(z,z)
        x=np.diag(x)
        y=np.diag(y)
        z=np.diag(z)

        if avoid_buildings:
            grid = self.filter_points(x, y, z,tolerance)
        return grid

    def gen_cart_grid(self,x,y,z,avoid_buildings=True,tolerance=0):
        '''
        Generates a cartesian grid of points from the input parameters. If desired the points that lie inside of
        buildings can be omitted.

        Parameters
        ----------
        x: array_like
              1-D arrays representing the coordinates of a grid.
        y: array_like
               1-D arrays representing the coordinates of a grid.
        z: array_like
               1-D arrays representing the coordinates of a grid.
        avoid_buildings: bool

        Returns
        -------
        grid: array_like
                2-D array representing the coordinates of the points [X,Y,Z]
        '''
        x,y = np.meshgrid(x,y)
        z,_ = np.meshgrid(z,z)
        x=np.diag(x)
        y=np.diag(y)
        z=np.diag(z)
        if avoid_buildings:
            grid = self.filter_points(x, y, z,tolerance)
        else:
            grid = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
            grid = grid.reshape(int(len(grid) / 3), -1)

        return grid

    def filter_points(self,x,y,z,tolerance=0,scale = 1):
        '''
        Filters out points which lie inside or in the vicinity of buildings. The value of tolerance acts as a buffer
        around the buildings where points are also filtered out.

        Parameters
        ----------
        x: array_like
        y: array_like
        z: array_like or float
        tolerance: float
        scale: float
            unused

        Returns
        -------
        grid: array_like
                2-D array representing the coordinates of the points [X,Y,Z]
        '''
        x = x.astype(float)
        y = y.astype(float)


        if isinstance(z, float) or isinstance(z, int):
            z = np.ones_like(x)*z
        else:
            z = z.astype(float)

        for building in self.buildings:
            mask = (x + tolerance > building.x_og) & (x - tolerance < building.x_og+building.x_extent) & \
                   (y + tolerance > building.y_og) & (y - tolerance < building.y_og+building.y_extent) & \
                   (z < building.z_extent)
            x[mask] = np.nan
            y[mask] = np.nan
            z[mask] = np.nan

        for i in range(len(x)):
            x[i], y[i]= rotate_origin_only(x[i],y[i],self.global_angle)

        grid = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        grid = grid[~np.isnan(grid)]
        grid = grid.reshape(int(len(grid)/3),-1)

        return grid

    def get_building_boundaries(self):
        '''
        Returns a list of the boundaries of the buildings

        Returns
        -------
        boundaries: list
        '''
        boundaries=[]
        for building in self.buildings:
            boundaries.append(building.boundaries)

        return boundaries

def intersects(s0,s1):
    '''
    Tests if 2 line segments are intersecting. Assumes line segments are stored in the format [(x0,y0),(x1,y1)].

    Parameters
    ----------
    s0: list of tuples
    s1: list of tuples

    Returns
    -------
    bool
    '''
    s1[0] = np.asarray(s1[0]).flatten()
    s1[1] = np.asarray(s1[1]).flatten()



    dx0 = s0[1][0]-s0[0][0]
    dx1 = s1[1][0]-s1[0][0]
    dy0 = s0[1][1]-s0[0][1]
    dy1 = s1[1][1]-s1[0][1]
    p0 = dy1*(s1[1][0]-s0[0][0]) - dx1*(s1[1][1]-s0[0][1])
    p1 = dy1*(s1[1][0]-s0[1][0]) - dx1*(s1[1][1]-s0[1][1])
    p2 = dy0*(s0[1][0]-s1[0][0]) - dx0*(s0[1][1]-s1[0][1])
    p3 = dy0*(s0[1][0]-s1[1][0]) - dx0*(s0[1][1]-s1[1][1])
    return (p0*p1<=0) & (p2*p3<=0)

def rotate_origin_only(x,y, angle):
    '''
    Only rotate a point around the origin (0, 0).

    Parameters
    ----------
        x (numpy.ndarray): X-components of input vectors.
        y (numpy.ndarray): Y-components of input vectors.
        angle (float): angle of rotation, in degrees

    Returns
    ----------

    '''
    angle = np.deg2rad(angle)
    xx = x * np.cos(angle) + y * np.sin(angle)
    yy = -x * np.sin(angle) + y * np.cos(angle)

    return xx, yy

def cost_func(angles,angle_cost=4):
    '''
    arbitrary cost function which punishes small angles: for visuals follow
    plt.figure()
    plt.plot(np.arange(91),(np.cos(np.deg2rad(np.arange(91)*4))+0.2)*((np.cos(np.deg2rad(np.arange(91)*4))+0.2)>0)*4)
    :param angles:

    Returns
    ----------
    cost: float
    '''
    cost = (np.cos(np.deg2rad(angles*4))+0.2)*((np.cos(np.deg2rad(angles*4))+0.2)>0)*angle_cost
    cost[cost<1] = 1
    cost[angles==0] = 1
    cost[angles==90]= 1

    return cost

def get_metangle(x, y):
    '''Get meteorological angle of input vector.

    Parameters
    ----------
        x (numpy.ndarray): X-components of input vectors.
        y (numpy.ndarray): Y-components of input vectors.

    Returns
    ----------
        (numpy.ma.core.MaskedArray): Meteorological angles.
    '''
    mask = np.logical_and(x == 0, y == 0)  # Vectors (0, 0) not valid.
    met_ang = ma.masked_array((90 - np.degrees(np.arctan2(x, y)) + 360) % 360, mask=mask,
                                  fill_value=np.nan)
    return met_ang

def optimize_grid(points,configuration,avoid_buildings=True,angle_cost=10):
    '''
    This function optimizes an input grid of points by minimizing the traveltime and angle between each point. Angles
    between points of ]0 - 25 and 65-90[ are punished in terms of traveltime and are avoided. Furthermore routes through
    buildings can be avoided aswell.

    Parameters
    ----------
    points: array_like
    configuration: object
    avoid_buildings: bool
    angle_cost: float

    Returns
    -------
    path
    '''
    if points.shape[1]==3:
        points2d = points[:, :2] # throwing away third dimension (z)
    else:
        points2d = points
    obstacle = np.zeros([len(points2d), len(points2d)])

    for i, point in enumerate(points2d):
        # first the angles between all points (nodes) are calculated
        angles = np.rad2deg(np.arcsin(np.abs(point - points2d)[:, 0] /
                                         (np.sqrt((point - points2d)[:, 0] ** 2 + (point - points2d)[:, 1] ** 2))))
        angles = np.nan_to_num(angles)

        # If buildings should be avoided in the paths between points, then every path (edge) between each point (node)
        # is checked if it intersects with a wall of a any building.
        if avoid_buildings:
            for j, node in enumerate(points2d):
                obstacle[i, j] = np.any([intersects([point, node], side) for building in configuration.get_building_boundaries()
                                         for side in building])

        # Then the distance between every point is calculated (pythagoras) and weighted with the angle (cost_func).
        # Acute angles are punished heavily.
        if i == 0:
            dist_all = np.hypot((point - points2d)[:, 0], (point - points2d)[:, 1]) * cost_func(angles, angle_cost)
        else:
            dist_all = np.vstack(
                [dist_all, np.hypot((point - points2d)[:, 0], (point - points2d)[:, 1]) * cost_func(angles, angle_cost)])
    # Distances of edges which intersect buildings are set to a high number so the traveling salesman algorithm does not
    # choose to go that path. However if no other path is available, it is still possible that the path intersects buildings.
    # One possible solution to this could be to add new points as intermediate step points to avoid buildings and acute
    # angles together.
    # Setting the distance to np.nan does not work. The tsp algorithm is not able to interpret nans properly. Another
    # solution is to completely disallow edges which intersect buildings. This could be implemented in the future.
    if avoid_buildings:
        dist_all[obstacle==1] = 1000000

    path = solve_tsp(dist_all, endpoints=(0, None))

    return points[path]

def rotate_via_numpy(xy, angle):
    """Use numpy to build a rotation matrix and take the dot product.
    
    Parameters
    ----------
    xy: array-like
    angle: float

    Returns
    ----------
    xy_transformed: array-like
    
    """
    # build rotation matrix
    if len(xy.flatten())==2:
        xy=[xy.astype(float)]
    xy_tranformed = np.copy(xy)
    radians = np.deg2rad(angle)
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])

    # rotate coordinates via j
    for i,point in enumerate(xy):
        m = np.dot(j, [point[0], point[1]])
        x_trans = m.T[0]
        y_trans = m.T[1]
        xy_tranformed[i] = np.stack([float(x_trans),float(y_trans)])

    return xy_tranformed



################################################################################
# A simple algorithm for solving the Travelling Salesman Problem
# Finds a suboptimal solution
# adapted from the tsp-solve2 package to remove the need to install a single package with pip
################################################################################




def optimize_solution(distances, connections, endpoints):
    """Tries to optimize solution, found by the greedy algorithm"""
    N = len(connections)
    path = restore_path(connections, endpoints)

    def ds(i, j):  # distance between ith and jth points of path
        pi = path[i]
        pj = path[j]
        if pi < pj:
            return distances[pj][pi]
        else:
            return distances[pi][pj]

    d_total = 0.0
    optimizations = 0
    for a in range(N - 1):
        b = a + 1
        for c in range(b + 2, N - 1):
            d = c + 1
            delta_d = ds(a, b) + ds(c, d) - (ds(a, c) + ds(b, d))
            if delta_d > 0:
                d_total += delta_d
                optimizations += 1
                connections[path[a]].remove(path[b])
                connections[path[a]].append(path[c])
                connections[path[b]].remove(path[a])
                connections[path[b]].append(path[d])

                connections[path[c]].remove(path[d])
                connections[path[c]].append(path[a])
                connections[path[d]].remove(path[c])
                connections[path[d]].append(path[b])
                path[:] = restore_path(connections, endpoints)

    return optimizations, d_total


def restore_path(connections, endpoints):
    """Takes array of connections and returns a path.
    Connections is array of lists with 1 or 2 elements.
    These elements are indices of teh vertices, connected to this vertex
    Guarantees that first index < last index
    """
    # when endpoints is None, then both start and end are not specified.
    start, end = endpoints or (None, None)
    # Now, if start, end or both are not specified - replace them with  replace unspecified start or end endpoints with the found ones
    need_revert = False
    is_loop = (start is not None) and (start == end)
    if start is None:
        if end is None:
            # find first node that only have one link.
            start = next(idx
                         for idx, conn in enumerate(connections)
                         if len(conn) == 1)
        else:
            # In this case - search from the end, then reverse the order
            start = end
            need_revert = True

    # ready to generate the path now.
    path = [start]
    prev_point = None
    cur_point = start
    # We know that path len should be the same as number of connection nodes, or one more if we are searching for a loop.
    # We already have one node, so need the rest
    for _ in range(len(connections) - (0 if is_loop else 1)):
        next_point = next(pnt for pnt in connections[cur_point]
                          if pnt != prev_point)
        path.append(next_point)
        prev_point, cur_point = cur_point, next_point
    if need_revert:
        return path[::-1]
    else:
        return path


def _assert_triangular(distances):
    """Ensure that matrix is left-triangular at least.
    """
    for i, row in enumerate(distances):
        if len(row) < i: raise ValueError(
            "Distance matrix must be left-triangular at least. Row {row} must have at least {i} items".format(
                **locals()))


def pairs_by_dist(N, distances):
    """returns list of coordinate pairs (i,j), sorted by distances; such that i < j"""
    # Sort coordinate pairs by distance
    indices = []
    for i in range(N):
        for j in range(i):
            indices.append(i * N + j)

    indices.sort(key=lambda ij: distances[ij // N][ij % N])
    return ((ij // N, ij % N) for ij in indices)


def solve_tsp(distances, optim_steps=3, pairs_by_dist=pairs_by_dist, endpoints=None):
    """Given a distance matrix, finds a solution for the TSP problem.
    Returns list of vertex indices.
    Guarantees that the first index is lower than the last

    :arg: distances : left-triangular matrix of distances. array of arrays
    :arg: optim_steps (int) number of additional optimization steps, allows to improve solution but costly.
    :arg: pairs_by_dist (function) an implementtion of the pairs_by_dist function. for optimization purposes.
    :arg: endpoinds : None or pair (int or None, int or None). Specifies start and end nodes of the path. None is unspecified.
    """
    N = len(distances)
    start, end = endpoints or (None, None)
    # When both points are specified, we are looking for a loop.
    is_loop = (start is not None) and (start == end)
    if start is not None and not (0 <= start < N): raise ValueError("Start point does not belong range")
    if end is not None and not (0 <= end < N): raise ValueError("Start point does not belong range")

    if N == 0: return []
    if N == 1:
        return [0, 0] if is_loop else [0]
    if N == 2 and is_loop:
        # Too short loops are easier to bypass here
        return [start, 1 - start, start]
    _assert_triangular(distances)

    # State of the TSP solver algorithm.
    node_valency = pyarray('i', [2]) * N  # Initially, each node has 2 sticky ends
    has_both_endpoints = (start is not None) and (end is not None)

    if not is_loop:
        # in a loop, all nodes have valency 2. Otherwise - start and end has 1
        if start is not None:
            node_valency[start] = 1
        if end is not None:
            node_valency[end] = 1

    # for each node, stores 1 or 2 connected nodes
    connections = [[] for i in range(N)]

    def join_segments(sorted_pairs):
        # segments of nodes. Initially, each segment contains only 1 node
        segments = [[i] for i in range(N)]

        def possible_edges():
            # Generate sequence of graph edges, that are possible and connect different segments.
            for ij in sorted_pairs:
                i, j = ij
                # if both start and end could have connections,
                #  and both nodes connect to a different segments:
                if node_valency[i] and node_valency[j] and \
                        (segments[i] is not segments[j]):
                    yield ij

        def connect_vertices(i, j):
            node_valency[i] -= 1
            node_valency[j] -= 1
            connections[i].append(j)
            connections[j].append(i)
            # Merge segment J into segment I.
            seg_i = segments[i]
            seg_j = segments[j]
            if len(seg_j) > len(seg_i):
                seg_i, seg_j = seg_j, seg_i
                i, j = j, i
            for node_idx in seg_j:
                segments[node_idx] = seg_i
            seg_i.extend(seg_j)

        def edge_connects_endpoint_segments(i, j):
            # return True, if given ede merges 2 segments that have endpoints in them
            # when this happens, search would terminate prematurely.
            # ALso works with the case when both endpoints are the same.
            si, sj = segments[i], segments[j]
            ss, se = segments[start], segments[end]
            return (si is ss) and (sj is se) or (sj is ss) and (si is se)

        # Take first N-1 possible edge. they are already sorted by distance
        edges_left = N - 1
        for i, j in possible_edges():
            if has_both_endpoints and edges_left != 1 and edge_connects_endpoint_segments(i, j):
                continue  # don't allow premature path termination
            connect_vertices(i, j)
            edges_left -= 1
            if edges_left == 0:
                break
        # if searching for a loop then close it
        if is_loop:
            _close_loop(connections)

    # invoke main greedy algorithm
    join_segments(pairs_by_dist(N, distances))

    # now call additional optiomization procedure.
    for passn in range(optim_steps):
        nopt, dtotal = optimize_solution(distances, connections, endpoints)
        if nopt == 0:
            break
    # restore path from the connections map (graph) and return it
    return restore_path(connections, endpoints=endpoints)


def _close_loop(connections):
    """Modify connections to close the loop"""
    i, j = (i for i, conn in enumerate(connections)
            if len(conn) == 1)
    connections[i].append(j)
    connections[j].append(i)
