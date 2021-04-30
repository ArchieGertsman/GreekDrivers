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
from joblib import Parallel, delayed
from pyproj import Geod
import geopandas
import nvector as nv
from nvector import rad


def bearing(df):
    """calculates and adds bearing column to dataframe
    Example usage:
        df = csv_to_df('sample.csv')
        df = bearing(df)
    """
    df['bearing'] = df \
        .groupby('id', as_index=False, group_keys=False) \
        .apply(__calc_bearings)
    return df


def nearest_graph_data(df, graph, mode='balltree'):
    """uses osmnx to find nearest node and edge data, calculates 
    progress along nearest edge as a ratio, and adds these features
    as columns to the dataframe. `mode` argument controls which method
    is used to compute nearest edges.
    Example usage:
        df = csv_to_df('sample.csv')
        g = ox.graph_from_...
        df = nearest_graph_data(df, g)
    """
    if mode == None:
        df = __apply_parallel(df, __construct_graph_data_cols(graph))
        return df
    elif mode == 'balltree':
        ret = ox.get_nearest_edges(graph, df['lon'], df['lat'], method='balltree')
    elif mode == 'kdtree':
        g_proj, df_proj = __proj(graph, df)
        ret = ox.get_nearest_edges(g_proj, df_proj['geometry'].x, df_proj['geometry'].y, method='kdtree', dist=100)
        df = df.drop('geometry', axis=1)
    else:
        raise ValueError('`mode` must be one of None, \'balltree\', or \'kdtree\'')
        return None

    df[['nearest_edge_start_node', 'nearest_edge_end_node']] = np.sort(ret[:,:2])
    df['edge_progress'] = df.apply(__edge_progress, axis=1, args=(graph,))
    return df


def direction(df):
    """adds column that determiens which direction the vehicle is moving along an edge.
    1 if moving from node with smaller id to node with larger id, 0 otherwise.
    Note: `nearest_graph_data` must have been run on this df, otherwise this will fail!
    Example usage:
        df = csv_to_df('sample.csv')
        g = ox.graph_from_...
        df = nearest_graph_data(df, g)
        df = direction(df)
    """
    df['dir'] = df \
        .groupby(['id', 'nearest_edge_start_node', 'nearest_edge_end_node'], 
            as_index=False, group_keys=False) \
        .apply(__calc_directions)
    return df


def vehicle_density(df):
    """returns a dataframe of the unique edges (nearest_edge_start_node and neares_edge_end_node pairs) 
    per direction (0 or 1) for edge progress intervals (in the range(0.0:0.9), 0.0 represents edge progress 
    between 0-10%, 0.1 represents edge progress between 10-20% and so on. 
    Node: df must have been processed by `direction` first. 
    Example usage: 
        df = csv_to_df(csv.file)
        g = ox.graph_from_...
        df = nearest_graph_data(df, g)
        df = direction(df)
        vehicle_density(df)
     """
    _,df["time_stamp"] = list(zip(*df.index))
    df['edge_progress_intervals'] = df                          \
        .groupby(['nearest_edge_start_node'])['edge_progress']  \
        .transform(lambda x: x-x%0.1)

    df2 = df                                            \
        .reset_index()                                  \
        .groupby([                                      \
            'nearest_edge_start_node',                  \
            'nearest_edge_end_node',                    \
            'dir',                                      \
            'edge_progress_intervals','time_stamp'])    \
        .agg({'id':['nunique']})
    return df2


def edge_average_speed(df):
    """returns a dataframe of the average speed of each edge (nearest_edge_start_node 
    and nearest_edge_end_node pairs) for both directions(0 or 1)
    Node: df must have been processed by `direction` first. 
    Example usage:
        df = Data('sample.csv').df
        g = ox.graph_from_...
        df = nearest_graph_data(df, g)
        df = direction(df)
        edge_average_speed(df)
     """
    _,df["time_stamp"] = list(zip(*df.index))
    df['edge_progress_intervals'] = df                          \
        .groupby(['nearest_edge_start_node'])['edge_progress']  \
        .transform(lambda x: x-x%0.1)                           \

    df2 = df                                \
        .reset_index()                      \
        .groupby([                          \
            'nearest_edge_start_node',      \
            'nearest_edge_end_node',        \
            'edge_progress_intervals',      \
            'dir','time_stamp'])['speed']   \
        .mean()
    
    return df2


def split_trajectories(df, size):
    """splits each vehicle's trajectory into smaller trajectories of fixed size,
    adding another dimension to the multiindex. Data is truncated to be a multiple
    of `size` in length. 
    Example usage:
        df = csv_to_df('sample.csv')
        df = split_trajectories(df, 3000)
    """
    return df.groupby('id', as_index=False, group_keys=False) \
            .apply(__split_vehicle, size)


def cross_track(df,graph):
    """computes the cross track distance of the vehicle at each timestamp
    Note: `nearest_graph_data` must have been run on this df, otherwise this will fail!
    Example usage:
        df = csv_to_df('sample.csv')
        g = ox.graph_from_...
        df = nearest_graph_data(df, g)
        df = cross_track(df)
    """
    return df.groupby(['nearest_edge_start_node', 'nearest_edge_end_node'], as_index=False, group_keys=False) \
        .apply(__calc_xtrack_dists, graph)

    
def edge_encoding(df):
    df_edge_list = df.reset_index()[['id','edge_id']].drop_duplicates()
    df_edge_list.set_index(['id','edge_id'],inplace = True)
    df_edge_list['edge_id'] = df_edge_list.index.get_level_values('edge_id') 
    df_edge_dummies = pd.get_dummies(df_edge_list)
    return df.join(df_edge_dummies)



# helper functions

def __apply_parallel(df, func, n=4):
    """parallelizes df.apply"""
    df_struct = dict(df.dtypes)
    idx_names = df.index.names
    retLst = Parallel(n_jobs=n)(delayed(func)(row) for _,row in df.iterrows())
    df = pd.concat(retLst, axis=1).T
    df.index.names = idx_names
    df = df.astype(df_struct)
    return df


def __bearing(c1, c2):
    """credit to https://bit.ly/3amjz0Q for bearing formula"""
    lat1,lon1 = c1
    lat2,lon2 = c2
    
    dL = lon2 - lon1
    x = cos(lon2) * sin(dL)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dL)
    return arctan2(x,y)


def __calc_bearings(df):
    """returns a multi-indexed dataframe of bearings at each timestep"""
    df1 = df
    df2 = df.shift(-1)

    c1 = (df1['lat'], df1['lon'])
    c2 = (df2['lat'], df2['lon'])
    df3 = __bearing(c1, c2)
    return df3


def __construct_graph_data_cols(graph):
    def aux(row):
        coord = (row['lat'],row['lon'])
        start, end, _ = ox.get_nearest_edge(graph, coord)
        if start > end:
            start, end = end, start
        row['nearest_edge_start_node'] = start
        row['nearest_edge_end_node'] = end
        row['edge_progress']  = __edge_progress(row, graph)
        return row
    return aux


def __edge_progress(row, graph):
    """calculates the progress of a vehicle along its nearest edge using the formula
    progress = (distance from start of edge to vehicle) / (length of edge)
    """
    edge_start_node = row['nearest_edge_start_node']
    edge_end_node = row['nearest_edge_end_node']

    lon_start, lat_start = graph.nodes[edge_start_node]['x'], graph.nodes[edge_start_node]['y']
    lon_end, lat_end = graph.nodes[edge_end_node]['x'], graph.nodes[edge_end_node]['y']
    lon_v, lat_v = row[['lon', 'lat']]

    geod = Geod(ellps='WGS84')
    _,_,a = geod.inv(lon_start, lat_start, lon_end, lat_end)
    _,_,b = geod.inv(lon_start, lat_start, lon_v, lat_v)
    return b/a


def __calc_directions(df):
    df1 = df
    df2 = df.shift(-1)
    df3 = (df1['edge_progress'] < df2['edge_progress']).astype(int)
    if len(df3) > 1:
        df3.iloc[-1] = df3.iloc[-2]
    return df3


def __truncate_to_multiple(n, m):
    return m * (n // m)

def __truncate_trajectory(traj, size):
    n = len(traj)
    new_len = __truncate_to_multiple(n, size)
    return traj[:new_len]

def __split_vehicle(df, size):
    """splits a vehicle trajectory into smaller trajectories of fixed size and removes
    the last (len(df) mod size) riws
    """
    df2 = df.copy()
    df2['traj'] = None
    df2.loc[::size, 'traj'] = np.arange(len(df2[::size]), dtype=int)
    df2['traj'].ffill(inplace=True)
    df2.set_index('traj', append=True, inplace=True)
    df2 = __truncate_trajectory(df2, size)
    df2 = df2.reorder_levels([0,2,1])
    return df2


def __proj(g, df):
    """project a graph and dataframe to the UTM zones in which their centroids lie"""
    WORLD_EPSG = 4326
    df_proj = geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df['lon'], df['lat']))
    df_proj.crs = WORLD_EPSG
    df_proj = ox.project_gdf(df_proj)
    g_proj = ox.project_graph(g)
    return g_proj, df_proj


def __calc_xtrack_dists(group, graph):
    """calculate the cross track distance for each row in a group, where all the
    rows in the group share the same nearest edge data
    """
    start_node, end_node = group[['nearest_edge_start_node', 'nearest_edge_end_node']].iloc[0]
    lon_start, lat_start = graph.nodes[start_node]['x'],graph.nodes[start_node]['y']
    lon_end, lat_end = graph.nodes[end_node]['x'],graph.nodes[end_node]['y']
    lon_v, lat_v = group['lon'], group['lat']

    start_pt = nv.lat_lon2n_E(rad(lat_start), rad(lon_start))
    end_pt = nv.lat_lon2n_E(rad(lat_end), rad(lon_end))
    v_pt = nv.lat_lon2n_E(rad(lat_v), rad(lon_v))

    group['xtrack_dist'] = nv.cross_track_distance((start_pt,end_pt), v_pt)
    return group