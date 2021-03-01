"""data.py
by: Archie Gertsman (arkadiy2@illinois.edu)
Project director: Richard Sowers
r-sowers@illinois.eduhttps://publish.illinois.edu/r-sowers/
Copyright 2019 University of Illinois Board of Trustees. All Rights Reserved. Licensed under the MIT license
"""

import numpy as np
import pandas as pd


def csv_to_df(csv_fname):
    """constructs a multi-indexed dataframe from a specified CSV file
    output format: 
        (id,time) -> [lat, lon, speed, lon_acc, lat_acc, type, traveled_d, avg_speed]
    """
    row_strs = __read_row_strs(csv_fname)
    dfs = [__row_str_to_df(row) for row in row_strs]
    return pd.concat(dfs)


def pickle_df(df, n=0, pickle_name='sample.pkl'):
    """saves first n id's in a pickle file"""
    if n == 0:
        n = df.index.levels[0].size
    
    df.loc[:n].to_pickle(pickle_name)

def df_from_coords(dataset): #creating gdf from max and min longitudes and latitudes from ampneuma dataset, dataset is expected to be created using csv_to_df
    max_lon = np.max(df["lon"])
    max_lat = np.max(df["lat"])
    min_lon = np.min(df["lon"])
    min_lat = np.min(df["lat"])
    return ox.geometries_from_bbox(max_lat,min_lat,max_lon,min_lon,tags={'building':True, 'landuse':True,'highway':True})



# helper functions

def __read_row_strs(csv_fname):
    with open(csv_fname, "r") as f:
        temp = f.readlines()

    return temp[1:] # exclude CSV header


def __row_str_to_df(row_str):
    header, data = __extract_parts(row_str)

    id = int(header[0].strip())
    timesteps = data[:,-1]
    mulidx = __create_mulidx(id, timesteps)

    data = data[:,:-1] # exclude time from data
    return __parts_to_df(mulidx, header, data)


def __extract_parts(row_str):
    # header features: id, type, traveled_d, avg_speed
    H = 4 # header length

    # data features: lat, lon, speed, lon_acc, lat_acc, time
    D = 6 # data length

    parts = row_str.strip().strip(";").split(";")
    header = parts[:H]
    data = np.array(parts[H:], dtype=np.float)
    data = data.reshape(-1, D)
    return header, data


def __create_mulidx(id, timesteps):
    id_arr = np.full(timesteps.shape, id)
    tups = list(zip(id_arr, timesteps))
    return pd.MultiIndex.from_tuples(tups, names=['id', 'time'])


def __parts_to_df(mulidx, header, data):
    col_names = ['lat', 'lon', 'speed', 'lon_acc', 'lat_acc']
    df = pd.DataFrame(data, columns=col_names, index=mulidx)
    df = df.assign(
        type=header[1].strip(),
        traveled_d=float(header[2]),
        avg_speed=float(header[3])
    )
    return df

  

