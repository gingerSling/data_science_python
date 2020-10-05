import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool, cv
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn import preprocessing
import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp


def parameter_settings_catboost():
    parameter_settings_catboost = {
    "tune_parameters": {
        "n_samples": 30,
        "num_threads": 1,
        "max_concurrent": 1
        },
    "parameters_types": {
        "weeks_to_use": "discrete",
        "agg_func": "categorical",
        "smoothing_window": "discrete",
        "days_to_use": "discrete",
        "learning_rate": "continuos",
        "iterations": "discrete",
        "depth": "discrete",
        "l2_leaf_reg": "discrete",
        "random_strength": "discrete"
        },
    "parameter_range": {
        "learning_rate": [0.0001, 1],
        "iterations": [600, 2500],
        "depth": [2, 7],
        "l2_leaf_reg": [0, 10],
        "random_strength": [1, 6]
    }
    }
    return parameter_settings_catboost

def var_type_sample(k, value_range, type_dict):
    var_type = type_dict[k]
    if var_type == 'categorical':
        return hp.choice(k, value_range)
    if var_type == 'continuos':
        return hp.uniform(k, value_range[0], value_range[1])
    if var_type == 'discrete':
        return hp.quniform(k, value_range[0], value_range[1], 1)

def cross_validation_catboost(X, y, model,n_folds, metric_to_use, loss_function, hp=None):
    model = model(**hp)
    metrics = []

    kf = KFold(n_splits=n_folds)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        data_matrix = Pool(data=X_train, label=y_train, cat_features=[49,50,51,52])
        model.fit(data_matrix,
          verbose=False)
        y_pred = model.predict(X_test)
        metric = metric_to_use(y_test, y_pred)
        print(metric)
        metrics.append(metric)
    metric_ave = np.mean(metrics)
    return metric_ave


def cross_validation_catboost_ray(config, reporter):
    X = pd.read_csv(config['non_hp']['data']['X'])
    X = X.drop(columns=['postcode_sector'])
    y = X.pop(config['non_hp']['data']['y'])
    cat_features = config['non_hp']['data']['cat_features']
    weights = config['non_hp']['data']['weights']
    weights = pd.DataFrame([1] * X.shape[0], index=X.index)
    metric_to_use = config['non_hp']['metric_to_use']

    parameter_dictionary = {k: v for k, v in config.items() if k not in
                                ['tune_parameters', 'parameters_types', 'parameter_range', 'model_settings', 'non_hp']}
    parameter_dictionary.update(config['model_settings'])

    Model = config['non_hp']['modelIns']
    model = Model(**parameter_dictionary)
    n_folds = config['non_hp']['n_folds']
    metrics = []
    kf = KFold(n_splits=n_folds)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        weights_train, weights_test = weights.loc[train_index], weights.loc[test_index]

        data_matrix = Pool(data=X_train, label=y_train, cat_features=cat_features, weight=weights_train)
        model.fit(data_matrix,
          verbose=False)
        y_pred = model.predict(X_test)
        metric = metric_to_use(y_test, y_pred)
        print(metric)
        metrics.append(metric)
    metric_ave = np.mean(metrics)
    reporter(
        metric_ave=metric_ave,
        done=True,
        )




def ray_tune_interface(data_settings, cross_validation_settings, parameter_settings, model_settings):
    cat_features = data_settings['cat_features']
    weights = data_settings['weights']
    data = data_settings['data']
    target = data_settings['target']
    
    model = model_settings.pop('model', None)

    n_folds = cross_validation_settings['n_folds']
    metric_to_use = cross_validation_settings['metric_to_use']
    mode = cross_validation_settings['mode']

    n_samples = parameter_settings['tune_parameters']['n_samples']
    num_threads = parameter_settings['tune_parameters']['num_threads']
    max_concurrent = parameter_settings['tune_parameters']['max_concurrent']
    space = ({k: var_type_sample(k, v, parameter_settings['parameters_types']) for k, v in parameter_settings['parameter_range'].items()})
    algo = HyperOptSearch(
        space,
        max_concurrent=max_concurrent,
        metric="metric_ave",
        mode=mode,
    )
    scheduler = ASHAScheduler(metric="metric_ave", mode=mode)
    config = {
        'num_samples': n_samples,
        'config': {
            'non_hp':{
                'data': {
                    'X': data,
                    'y': target,
                    'cat_features': cat_features,
                    'weights': weights,
                },
                'modelIns': model,
                'n_folds': n_folds,
                'metric_to_use': metric_to_use,
            }
        }
    }
    config['config'].update(parameter_settings)
    config['config']['model_settings'] = model_settings
    ray_experiment = tune.run(cross_validation_catboost_ray,
                    resources_per_trial={"gpu": 1},
                    search_alg=algo,
                    scheduler=scheduler,
                    keep_checkpoints_num=0,
                    verbose=1,
                    **config)
    results = ray_experiment.dataframe(metric="metric_ave", mode="min")
    results.to_csv('results_experiment.csv')

