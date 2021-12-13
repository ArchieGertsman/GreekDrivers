from modeling_helpers import *

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Objective:
    """instances are meant to be passed into optuna's study.optimize"""

    def __init__(self, df_agg_path, model_generation_dict, metric, splitter_obj):
        """
        df_agg_path : str
            path of .pkl file containing df_agg
        model_generation_dict : (trial) -> dict
            function that constructs a dict based on an optuna trial, which
            contains the recipes for how optuna should suggest a model in
            that trial
        metric : classification metric from sklearn.metrics
            the metric by which to score the model
        splitter_obj : splitter class from sklearn.model_selection
            the train/test splitting scheme
        """
        self.df_agg = pd.read_pickle(df_agg_path)
        self.model_generation_dict = model_generation_dict
        self.metric = metric
        self.splitter_obj = splitter_obj
        

    def __call__(self, trial):
        """performs one trial of optimization. All model suggestions are abstracted
        away by the self.model_generation_dict function, as specified during instantiation
        """
        d = self.model_generation_dict(trial)
        model_name = trial.suggest_categorical('model_name', list(d.keys()))
        model_dict = d[model_name]

        def __suggest(param_name, v):
            suggest_func, suggest_params = v
            return suggest_func(name=f'{model_name}_{param_name}', **suggest_params)

        params = {k: __suggest(k,v) for k,v in model_dict['trial_params'].items()}
        if 'other_params' in model_dict:
            params |= model_dict['other_params']

        model = (model_dict['ctor'])(**params)
        model = Pipeline([
            ('scaler', StandardScaler()), 
            (model_name, model)])

        workflow_n_jobs = model_dict['workflow_n_jobs'] if 'workflow_n_jobs' in model_dict else 1
        
        accuracy,_ =  workflow(
            df_agg        = self.df_agg, 
            model         = model, 
            splitter_obj  = self.splitter_obj,
            metric        = self.metric,
            balance_train = 'by_road',
            balance_test  = True,
            n_jobs        = workflow_n_jobs
        )
        
        return accuracy  