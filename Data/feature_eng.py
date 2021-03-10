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
    bearing_list = [__calc_bearings_for_id(df, id) for id in df.index.unique(level=0)]
    df['bearing'] = pd.concat(bearing_list)
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
    dir_list = [__calc_dirs_for_id(df, id) for id in df.index.unique(level=0)]
    df['dir'] = pd.concat(dir_list)
    return df


def gdf_from_coords(dataset): #creating gdf from max and min longitudes and latitudes from ampneuma dataset, dataset is expected to be created using csv_to_df
    max_lon = np.max(df["lon"])
    max_lat = np.max(df["lat"])
    min_lon = np.min(df["lon"])
    min_lat = np.min(df["lat"])
    return ox.geometries_from_bbox(max_lat,min_lat,max_lon,min_lon,tags={'building':True, 'landuse':True,'highway':True})



# helper functions

def __bearing(c1, c2):
    """credit to https://bit.ly/3amjz0Q for bearing formula"""
    lat1,lon1 = c1
    lat2,lon2 = c2
    
    dL = lon2 - lon1
    x = cos(lon2) * sin(dL)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dL)
    return arctan2(x,y)


def __calc_bearings_for_id(df, id):
    """returns a multi-indexed dataframe of bearings at each timestep for vehicle with specified ID"""
    df1 = df.loc[id]
    df1 = df1.set_index(pd.Index(range(0,len(df1.index))))
    df2 = df1.set_index(df1.index - 1)
    df2 = df2.drop(-1)

    c1 = (df1['lat'], df1['lon'])
    c2 = (df2['lat'], df2['lon'])
    df3 = __bearing(c1, c2)
    df3.index = df.index[df.index.isin([id], level=0)]
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


def __calc_dirs_for_id(df, id):
    df1 = df.loc[id]
    df1 = df1.set_index(pd.Index(range(0,len(df1.index))))
    df2 = df1.set_index(df1.index - 1)
    df2 = df2.drop(-1)
    df2.at[len(df1)-1] = None

    df3 = (df1['edge_progress'] < df2['edge_progress']).astype(int)
    df3.iloc[-1] = df3.iloc[-2]
    df3.index = old_idx = df.index[df.index.isin([id], level=0)]
    return df3