import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import multiprocessing


def downsample(df, window, overlap, agg_dict, parallel=True, min_speed_ratio=None):
    """downsamples each (id,road) pair in the dataframe with overlap
    between consecutive windows
    
    Parameters
    ----------
    df : pd.DataFrame
        original DataFrame indexed by (id,road,time)
    window : int
        size of window to aggregate over, in number of rows
    overlap : float in [0,1)
        proportion of overlap between consecutive windows. E.g
        if `window`=50 and `overlap`=0.2, then consecutive 
        windows will share 50*0.2 = 10 rows. With `overlap`=0.0,
        this function behaves like the pandas `resample` function
        applied to each (id,road) pair.
    agg_dict : dict
        specifies which features to keep, and which function(s)
        to use for downsampling each feature. E.g.
            `agg_dict` = {..., 'speed': ['mean', 'std'], ...}
        signifies that the returned dataframe will contain
        two columns for speed: one downsampled using `mean`
        and the other using `std`. Note: only `mean` and `std`
        are allowed.
    parallel : bool, optional
        indicates whether or not to parallelize downsampling among
        (id,road) pairs
    
    Returns
    -------
    df_agg : pd.DataFrame
        downsampled dataframe indexed by (id,road). Index values
        are not unique, i.e. for each (id,road) pair there are
        numerous rows, each corresponding to a window's aggregation.
    """
    assert(__extract_agg_funcs(agg_dict) == set({'mean','std'}))
    
    df_cpy = df.copy()
    
    if min_speed_ratio is not None:
        df_cpy['speed_bool'] = (df_cpy.speed>0).astype(int)
        agg_dict['speed_bool'] = ['mean']
    
    step = int((1-overlap)*window)
    df_drop_type = df_cpy.drop('type', axis=1)
    
    if parallel:
        df_agg = __groupby_apply_parallel(df_drop_type, ['id','road'], 
                   __downsample_group, window, step)
    else:
        df_agg = df_drop_type.groupby(['id','road']) \
            .apply(__downsample_group, window, step)

    df_agg.columns = __downsample_cols(df_drop_type.columns)
    df_agg = df_agg[__extract_feature_list(agg_dict)]
    df_agg = __append_type_column(df_agg, df)
    
    if min_speed_ratio is not None:
        df_agg = df_agg[df_agg.speed_bool_mean >= 0.75]
        df_agg.drop('speed_bool_mean', axis=1, inplace=True)
    
    return df_agg


def train_test_split_vehicles(df, test_size, balance_train=True, balance_test=True):
    """splits the data into a train and test set, where the test set
    has `test_class_size` vehicles from each class"""
    ids_train,ids_test,types_test = \
        __train_test_split_ids(df, test_size, balance_test)
    
    df_train = __select_by_ids(df, ids_train)
    df_test = __select_by_ids(df, ids_test)
    
    if balance_train:
        """ the number of cars and taxis in each edge is balanced"""
        df_train = __balance_roads(df_train)
    
    return __split_X_y(df_train), __split_X_y(df_test)


def accuracy(model, X, y, metric=accuracy_score):
    """measures the accuracy of a trained model on test data by 
    a specified metric. Uses voting.
    """
    y = y.groupby('id').first()
    
    y_hat_p = model.predict_proba(X)[:,0]
    y_hat_p = pd.DataFrame(index=X.index, data=y_hat_p, columns=['type'])
    y_hat_p = y_hat_p.groupby(['id','road']).agg('mean')
    y_hat_p = y_hat_p.groupby('id').agg('mean')
    y_hat = __vote(y_hat_p, model.classes_)

#     y_hat_p = model.predict(X)
#     y_hat_p = pd.DataFrame(index=X.index, data=y_hat_p, columns=['type'])
#     y_hat_p = y_hat_p.groupby(['id','road']).agg(lambda x: x.mode()[0])
#     y_hat_p = y_hat_p.groupby('id').agg(lambda x: x.mode()[0])
#     y_hat = y_hat_p

    return metric(y, y_hat)


def balance_road(road):
    return __balance_road(road)



    

"""------------------"""
""" HELPER FUNCTIONS """
"""------------------"""





""" downsample """


def __groupby_apply_parallel(df, by, func, *args):
    g = df.groupby(by)
    idx_names = df.index.names
    def __temp_func(func, name, grp, *args):     
        return func(grp, *args), name
    
    lst,idx = zip(*Parallel(n_jobs=multiprocessing.cpu_count())
        (delayed(__temp_func)(func, name, grp, *args) for name,grp in g))
    df2 = pd.concat(lst, keys=idx)
    df2.index.names = idx_names + ['level_2']
    return df2


def __downsample_group(grp, window, step):
    """downsamples an individual (id,road) pair"""
    windows = [grp.iloc[i:i+window] for i in range(0, (len(grp)-window), step)]
    if len(windows)==0:
        return None
    windows = np.array(windows)
    aggs = (windows.mean(axis=1), windows.std(axis=1))
    agg = np.concatenate(aggs, axis=1)
    agg = pd.DataFrame(agg)
    return agg


def __extract_agg_funcs(agg_dict):
    """returns set of aggregation functions specified in `agg_dict`"""
    agg_funcs = [val for vals in agg_dict.values() for val in vals]
    agg_funcs = set(agg_funcs)
    return agg_funcs


def __downsample_cols(original_cols):
    """recovers the names of the columns after downsampling"""
    return ['_'.join([col,agg]) for agg in ['mean','std'] for col in original_cols]


def __extract_feature_list(agg_dict):
    """returns the list of features to keep from a downsampled dataframe.
    It is a sublist of what `downsample_cols` returns
    """
    return ['_'.join([key,val]) for key,vals in agg_dict.items() for val in vals]


def __append_type_column(df_agg, df_original):
    """append the 'type' column from `df_original` to `df_agg`"""
    vehicle_types = df_original.type.groupby(['id','road']).first()
    return df_agg.reset_index(-1, drop=True).join(vehicle_types)





""" train_test_split  """


def __train_test_split_ids(df_agg, test_size, balance_test):
    id_type_map = df_agg.groupby('id').type.first()
    ids,types = id_type_map.index.values,id_type_map.values
    ids_train,ids_test,_,types_test = train_test_split(ids, types, test_size=test_size, stratify=types)
    if balance_test:
        ids_test = __balance_ids(id_type_map.loc[ids_test])
    return ids_train,ids_test,types_test


def __select_by_ids(df_agg, ids):
    return df_agg[df_agg.index.get_level_values('id').isin(ids)]


def __balance_ids(id_type_map):
    id_type_map = pd.DataFrame(id_type_map)
    g = id_type_map.groupby('type', group_keys=False)
    return g.apply(lambda group: group.sample(g.size().min())).index.values


def __balance_roads(df_agg):
    df_agg = df_agg.groupby('road').apply(__balance_road)
    return df_agg.reorder_levels((1,0))


def __balance_road(road):
    """balances the number of cars and taxis in a road by resampling
    smaller class
    """
    class_counts = road.groupby('id').first().type.value_counts()
    n_resample = class_counts.max() - class_counts.min()
    road.reset_index('road', inplace=True, drop=True)
    idx_resample = __resample_idx(road, class_counts.idxmin(), n_resample)
    resample = road.loc[idx_resample]
    return pd.concat([road,resample])
    
    
def __resample_idx(road, vehicle_class, n_resample):
    """resamples indices from class `vehicle_class`, `n_resample` times"""
    idx_resample = road[road.type==vehicle_class] \
        .index \
        .unique() \
        .to_series() \
        .sample(n_resample, replace=True) \
        .values
    return idx_resample


def __split_X_y(df_agg):
    """splits labelled data into unlabelled data and labels"""
    return df_agg.drop('type', axis=1), df_agg.type





""" accuracy """

def __extreme(x):
    return x[np.abs((x-0.5)).argmax()]

def __vote(y_hat_p, classes):
    return y_hat_p.type.map(lambda x: classes[0] if x>=0.5 else classes[1])