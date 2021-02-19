"""feature_eng.py
by: Archie Gertsman (arkadiy2@illinois.edu)
Project director: Richard Sowers
r-sowers@illinois.eduhttps://publish.illinois.edu/r-sowers/
Copyright 2019 University of Illinois Board of Trustees. All Rights Reserved. Licensed under the MIT license
"""

import numpy as np
import pandas as pd
from numpy import arctan2, sin, cos


def bearing(df):
    """calculates and adds bearing column to dataframe
    Example usage:
        df = csv_to_df('sample.csv')
        df = bearing(df)
    """
    bearing_list = [__calc_bearings_for_id(df, id) for id in df.index.unique(level=0)]
    df['bearing'] = pd.concat(bearing_list)
    return df



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
    df_1 = df.loc[id]
    df_1 = df_1.set_index(pd.Index(range(0,len(df_1.index))))
    df_2 = df_1.set_index(df_1.index - 1)
    df_2 = df_2.drop(-1)

    old_idx = df.index[df.index.isin([id], level=0)]

    c1 = (df_1['lat'], df_1['lon'])
    c2 = (df_2['lat'], df_2['lon'])
    df = __bearing(c1, c2)
    df.index = old_idx
    return df