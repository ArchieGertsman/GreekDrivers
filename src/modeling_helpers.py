import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def downsample(df, window, overlap, agg_dict):
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
    
    Returns
    -------
    df_agg : pd.DataFrame
        downsampled dataframe indexed by (id,road). Index values
        are not unique, i.e. for each (id,road) pair there are
        numerous rows, each corresponding to a window's aggregation.
    """
    assert(__extract_agg_funcs(agg_dict) == set({'mean','std'}))
        
    step = int((1-overlap)*window)
    df_drop_type = df.drop('type', axis=1)
    
    df_agg = df_drop_type.groupby(['id','road']) \
        .apply(__downsample_group, window, step)
    df_agg.columns = __downsample_cols(df_drop_type.columns)
    df_agg = df_agg[__extract_feature_list(agg_dict)]
    df_agg = __append_type_column(df_agg, df)
    return df_agg


def train_test_split_vehicles(df_agg, test_class_size):
    """splits the data into a train and test set, where the test set
    has `test_class_size` vehicles from each class"""
    df_agg_test = __sample_test_set(df_agg, test_class_size)
    df_agg_train = __construct_train_set(
        df_agg, df_agg_test.index.get_level_values('id'))
    return df_agg_train, df_agg_test


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

    return metric(y, y_hat)


def split_X_y(df_agg):
    """splits labelled data into unlabelled data and labels"""
    return df_agg.drop('type', axis=1), df_agg.type



    

"""------------------"""
""" HELPER FUNCTIONS """
"""------------------"""





""" downsample """

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


# test set

def __sample_test_set(df_agg, class_size):
    """selects `class_size` vehicles from each class at random and
    combines them into a test set
    """
    df_reset = df_agg.reset_index('road')

    df_list = []
    for vehicle_class in ['Car','Taxi']:
        idx = __sample_class_idx(df_reset, vehicle_class, class_size)
        df_list.append(df_reset.loc[idx])

    df_agg_test = pd.concat(df_list).set_index('road', append=True)
    return df_agg_test


def __sample_class_idx(df, vehicle_class, class_size):
    """samples `class_size` indices from class `vehicle_class`.
    `df` must be indexed only by id.
    """
    idx_sample = df[df.type==vehicle_class] \
        .index \
        .unique() \
        .to_series() \
        .sample(class_size) \
        .values
    return idx_sample


# train set

def __construct_train_set(df_agg, test_ids):
    """balances the number of vehicles on each road by resampling the
    smaller class. Only selects vehicles that were not chosen to be
    in the test set
    """
    df_agg_train = df_agg.drop(index=test_ids, level='id')
    df_agg_train = df_agg_train.groupby('road').apply(__balance_road)
    df_agg_train.index = df_agg_train.index.reorder_levels((1,0))
    return df_agg_train


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





""" accuracy """

def __vote(y_hat_p, classes):
    return y_hat_p.type.map(lambda x: classes[0] if x>=0.5 else classes[1])