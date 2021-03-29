"""feature_eng.py
by: Archie Gertsman (arkadiy2@illinois.edu)
Project director: Richard Sowers
r-sowers@illinois.eduhttps://publish.illinois.edu/r-sowers/
Copyright 2019 University of Illinois Board of Trustees. All Rights Reserved. Licensed under the MIT license
"""

import numpy as np
import pandas as pd
from numpy import arctan2, sin, cos, sqrt, radians
import osmnx as ox


def bearing(df):
    """calculates and adds bearing column to dataframe
    Example usage:
        df = csv_to_df('sample.csv')
        df = bearing(df)
    """
    df['bearing'] = \
        df.groupby('id', as_index=False, group_keys=False) \
        .apply(__calc_bearings)
    return df


def nearest_graph_data(df, graph):
    """uses osmnx to find nearest node and edge data, calculates 
    progress along nearest edge as a ratio, and adds these features
    as columns to the dataframe
    Example usage:
        df = csv_to_df('sample.csv')
        graph = ox.graph_from_address('address_here', network_type='drive') 
        df = nearest_graph_data(df, graph)
    """
    df['nearest_node'],             \
    df['nearest_edge_start_node'],  \
    df['nearest_edge_end_node'],    \
    df['edge_progress']             \
        = zip(*df.apply(__construct_graph_data_cols(graph), axis=1))
    return df


def direction(df):
    """adds column that determiens which direction the vehicle is moving along an edge.
    1 if moving from node with smaller id to node with larger id, 0 otherwise.
    Note: `nearest_graph_data` must have been run on this df, otherwise this will fail!
    Example usage:
        df = csv_to_df('sample.csv')
        df = direction(df)
    """
    df['dir'] = \
        df.groupby(
            ['id', 'nearest_edge_start_node', 'nearest_edge_end_node'], 
            as_index=False, group_keys=False) \
        .apply(__calc_directions)
    return df


def vehicle_density(df):
    df = _calc_vehicle_density(df)
    return df

# helper functions

def _calc_vehicle_density(df):
    """returns a dataframe of the unique edges (nearest_edge_start_node and neares_edge_end_node pairs) per direction (0 or 1) for edge progress intervals (in the          range(0.0:0.9), 0.0 represents edge progress between 0-10%, 0.1 represents edge progress between 10-20% and so on. 
        df must have been processed by `direction` first. Example usage: 
        df = csv_to_df(csv.file)
        graph = ox.graph_from_address('Athens, Municipality of Athens, Regional Unit of Central Athens, Attica, 10667, Greece', network_type='drive')  
        df = nearest_graph_data(df,graph)
        df = direction(df)
        vehicle_densities = _calc_vehicle_density(df)
     """
    df['edge_progress_intervals'] = df.groupby(['nearest_edge_start_node'])['edge_progress'].transform(lambda x: x-x%0.1)
    df2 = df.reset_index().groupby(['nearest_edge_start_node','nearest_edge_end_node','dir','edge_progress_intervals']).agg({'id':['nunique']})
    return df2


def _calc_edge_average_speed(df):
    """returns a dataframe of the average speed of each edge (nearest_edge_start_node and nearest_edge_end_node pairs) for both directions(0 or 1)
        df = Data('sample.csv').df
        graph = ox.graph_from_address('Athens, Municipality of Athens, Regional Unit of Central Athens, Attica, 10667, Greece',           network_type='drive')  
        df = nearest_graph_data(df,graph)
        df = direction(df)
        edge_average_speed = _calc_edge_average_speed(df)
     """
    df['edge_progress_intervals'] = df.groupby(['nearest_edge_start_node'])['edge_progress'].transform(lambda x: x-x%0.1)
    df2 = df.reset_index().groupby(['nearest_edge_start_node','nearest_edge_end_node','edge_progress_intervals','dir'])['speed'].mean()
    return df2


def __bearing(c1, c2):
    """credit to https://bit.ly/3amjz0Q for bearing formula"""
    lat1,lon1 = c1
    lat2,lon2 = c2
    
    dL = lon2 - lon1
    x = cos(lon2) * sin(dL)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dL)
    return arctan2(x,y)


def __calc_bearings(df):
    """returns a multi-indexed dataframe of bearings at each timestep for vehicle with specified ID"""
    df1 = df
    df2 = df.shift(-1)

    c1 = (df1['lat'], df1['lon'])
    c2 = (df2['lat'], df2['lon'])
    df3 = __bearing(c1, c2)
    return df3


def __construct_graph_data_cols(graph):
    def aux(row):
        coord = (row['lat'],row['lon'])
        nn = ox.get_nearest_node(graph, coord, method='euclidean')
        start, end, _ = ox.get_nearest_edge(graph, coord)
        if start > end:
            start, end = end, start
        edge_prog = __edge_progress(graph, start, end, coord)
        return nn, start, end, edge_prog
    return aux


def __edge_progress(graph, edge_start_node, edge_end_node, v_coord):
    start_coord = graph.nodes[edge_start_node]['y'], graph.nodes[edge_start_node]['x']
    end_coord = graph.nodes[edge_end_node]['y'], graph.nodes[edge_end_node]['x']

    a = __euc_dist(start_coord, end_coord)
    b = __euc_dist(start_coord, v_coord)
    return b/a


def __euc_dist(coord0, coord1):
    EARTH_RADIUS = 6373

    lat0, lon0 = coord0
    lat1, lon1 = coord1

    lat0, lon0 = radians(lat0), radians(lon0)
    lat1, lon1 = radians(lat1), radians(lon1)

    dlat = lat1 - lat0
    dlon = lon1 - lon0

    a = sin(dlat/2)**2 + cos(lat0) * cos(lat1) * sin(dlon/2)**2
    c = 2 * arctan2(sqrt(a), sqrt(1-a))
    return EARTH_RADIUS*c


def __calc_directions(df):
    df1 = df
    df2 = df.shift(-1)
    df3 = (df1['edge_progress'] < df2['edge_progress']).astype(int)
    if len(df3) > 1:
        df3.iloc[-1] = df3.iloc[-2]
    return df3
