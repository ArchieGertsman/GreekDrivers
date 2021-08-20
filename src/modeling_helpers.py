import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.base import clone
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


def workflow(df_agg, model, splitter_obj, metric, metric_kwargs={}, balance_train=None, balance_test=True):
    """trains a model on a downsampled dataframe according to a splitting scheme
    and returns the mean & std accuracy according to a metric
    
    Parameters
    ----------
    df_agg : pd.DataFrame
        downsampled dataframe on which model will be trained
    model : sklearn estimator instance
        model that will be trained on `df_agg`
    splitter_obj : splitter class from sklearn.model_selection
        the train/test splitting scheme
    metric : classification metric from sklearn.metrics
        the metric by which to score the model
    metric_kwargs : dict, optional
        dict of named arguments that will be passed to `metric`
    balance_train : {'overall', 'by_road'} or None, optional
        `None` : 
            don't balance the training sets
        `'overall'` : 
            balance the classes in the training sets by 
            number of rows, resampling the smaller class
        `'by_road'` : 
            balance the classes for each road in the
            training set, resampling the smaller class
    balance_test : bool, optional
        if `True` then balances the classes in the test sets
        by number of vehicle ids, downsampling the larger class
    
    Returns
    -------
    acc_stats : 2-tupe of floats
        pair of mean and std accuracy
    """
    ids,labels = __ids_labels(df_agg)
    scores = []
    for train_idx, test_idx in splitter_obj.split(ids, labels):
        ids_train, ids_test = ids[train_idx], ids[test_idx]
        labels_train, labels_test = labels[train_idx], labels[test_idx]
        
        df_train = __select_by_ids(df_agg, ids_train)
        if balance_train is not None:
            df_train = __balance_train(df_train, method=balance_train)
        X_train,y_train = __split_X_y(df_train)

        if balance_test:
            ids_test = __balance_ids(ids_test,labels_test)
        df_test = __select_by_ids(df_agg, ids_test)
        X_test,y_test = __split_X_y(df_test)

        model = clone(model)
        model.fit(X_train, y_train)
        score = __score(model, X_test, y_test, metric, metric_kwargs)
        print(score)
        scores += [score]
    scores = np.array(scores)
    score_stats = scores.mean(axis=0), scores.std(axis=0)
    return score_stats





"""------------------"""
""" HELPER FUNCTIONS """
"""------------------"""





""" downsample """


def __groupby_apply_parallel(df, by, func, *args):
    """parallelizes `df.groupby(by).apply(func, args)` across groups"""
    g = df.groupby(by)
    idx_names = df.index.names
    def __temp_func(func, name, grp, *args):     
        return func(grp, *args), name
    
    lst,idx = zip(*Parallel(n_jobs=multiprocessing.cpu_count())
        (delayed(__temp_func)(func, name, grp, *args) for name,grp in g))
    df2 = pd.concat(lst, keys=idx)
    df2.index.names = idx_names + ['level_extra']
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

def __ids_labels(df_agg):
    """fetches array of ids and corresponding array of vehicle types"""
    id_label_map = df_agg.groupby('id').type.first()
    ids, labels = id_label_map.index.values, id_label_map.values
    return ids, labels


def __select_by_ids(df_agg, ids):
    """selects all rows from `df_agg` that are indexed by an id in `ids`"""
    return df_agg[df_agg.index.get_level_values('id').isin(ids)]


def __balance_ids(ids, labels):
    """balances number of ids from each class by downsampling larger class"""
    df_id_label = pd.DataFrame(data=np.array([ids,labels]).T, columns=['id','label'])
    g = df_id_label.groupby('label', group_keys=False)
    return g.apply(lambda group: group.sample(g.size().min())).id.values


def __balance_train(df_train, method):
    if method == 'by_road':
        return __balance_roads(df_train)
    elif method == 'overall':
        return __balance_overall(df_train)
    else:
        raise Exception('invalid balancing method')


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


def __balance_overall(df):
    df_agg = df.reset_index()
    class_counts = df_agg.type.value_counts()
    n_resample = class_counts.max() - class_counts.min()
    idx_resample = __resample_idx(df_agg, class_counts.idxmin(), n_resample)
    resample = df_agg.loc[idx_resample]
    df_balanced = pd.concat([df_agg,resample])
    return df_balanced.set_index(['id','road'])
    
    
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

def __score(model, X, y, metric=accuracy_score, metric_kwargs={}):
    """measures the accuracy of a trained model on test data by 
    a specified metric. Uses voting.
    """
    y = y.groupby('id').first()
    
    y_score = model.predict_proba(X)[:,1]
    y_score = pd.DataFrame(index=X.index, data=y_score, columns=['type'])
    y_score = y_score.groupby(['id','road']).agg('mean')
    y_score = y_score.groupby('id').agg('mean')
    y_hat = y_score \
        if metric==roc_auc_score \
        else __predict(y_score, model.classes_)

    return metric(y, y_hat, **metric_kwargs)


def __extreme(x):
    """returns element from array of probabilities `x` that is 
    farthest from 0.5
    """
    return x[np.abs((x-0.5)).argmax()]


def __predict(y_score, classes, threshold=0.5):
    """predicts classes from scores based on threshold"""
    return y_score.type.map(lambda x: classes[0] if x<threshold else classes[1])