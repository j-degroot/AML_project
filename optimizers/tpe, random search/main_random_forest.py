from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor

from hyperopt.pyll import scope
from hyperopt import hp, tpe, rand, fmin, Trials, STATUS_OK
import json
import numpy as np
import pickle
import csv


# load csv data
# TODO: loading data from csv files
# half data ofr training and second half for testing
# preprocessing - depends on how the data from each of us look like
X_train, y_train = []
X_test, y_test = []
y_train = []
y_test = []

print("Image Data Shape", len(X_train), len(X_train[0]))
print("Image test Data Shape", len(X_test), len(X_test[0]))
print("Target Data Shape", len(y_train))
print("Target test Data Shape", len(y_test))



# define space for random search on Random Forest regression
space = {'min_samples_split': hp.uniform('min_samples_split', 2, 10),
         'n_estimators': hp.uniform('n_estimators', 10, 2000),
         'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2'])}



def random_forest_acc(params):

    rf = RandomForestRegressor(min_samples_split=params['min_samples_split'],
                               n_estimators=params['n_estimators'],
                               max_features=params['max_features'])

    rf.fit(X_train, y_train)

    acc = rf.score(X_test, y_test)
    return acc


def obj_func(params):
    acc = random_forest_acc(params)
    loss = 1 - acc
    return {'loss': loss, 'status': STATUS_OK}


if __name__ == "__main__":
    """
    perform random search optimization on Random Forest Regressor
    to find best parameters for Random Forest as surrogate benchmark model
    """
    trials = Trials()
    best_params = fmin(fn=obj_func,
                    space=space,
                    algo=rand.suggest,
                    max_evals=100,
                    trials=trials)

    print("Best parameters:", best_params)
    print(trials.best_trial['result']['loss'])

    loss = trials.losses()
    val = trials.vals
    val['loss'] = loss
    print(val)

    with open('random_forest_best_params.json', 'w') as f:
        f.write(json.dumps({"Loss": trials.best_trial['result']['loss'],
                            "Best params": best_params}))

    filename = 'csv_data/random_forest.csv'
    header = ['min_samples_split', 'n_estimators', 'max_features', 'loss']
    values = (val.get(key, []) for key in header)
    data = (dict(zip(header, row)) for row in zip(*values))
    with open(filename, 'w') as f:
        wrtr = csv.DictWriter(f, fieldnames=header)
        wrtr.writeheader()
        wrtr.writerows(data)

