import sys
sys.path.append('../src/')
sys.path.append('../data/')
from modeling_helpers import *
from tuning_objective import Objective

import optuna
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

def __log(suggest_func, low, high):
    return (suggest_func, {'low':low, 'high':high, 'log':True})

def __step(suggest_func, low, high, step=1):
    return (suggest_func, {'low':low, 'high':high, 'step':step})

        
if __name__ == "__main__":
    study_name = "best_classifier3"
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name, 
        storage=storage_name,
        load_if_exists=True)

    model_generation_dict = lambda trial: {
        'gbm': {
            'ctor': GradientBoostingClassifier,
            'trial_params': {
                'learning_rate':    __log (trial.suggest_float, 0.05, 0.3),
                'n_estimators':     __step(trial.suggest_int,   100, 300, step=50),
                'max_depth':        __step(trial.suggest_int,   4, 15),
                'max_features':     __step(trial.suggest_int,   4, 13)
            },
            'workflow_n_jobs': 12
        },
        'rf': {
            'ctor': RandomForestClassifier,
            'trial_params': {
                'n_estimators':     __step(trial.suggest_int, 50, 200, step=50),
                'max_depth':        __step(trial.suggest_int, 2, 7),
                'max_features':     __step(trial.suggest_int, 4, 7)
            },
            'other_params': {
                'n_jobs': 8
            }
        }
    }

    objective = Objective(
        df_agg_path             = '../data/df_agg_50.pkl',
        model_generation_dict   = model_generation_dict,
        metric                  = accuracy_score, 
        splitter_obj            = RepeatedStratifiedKFold(5,3)
    )

    study.optimize(objective, n_trials=250)