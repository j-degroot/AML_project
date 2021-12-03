from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor

from hyperopt.pyll import scope
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import json
import numpy as np
import pickle
import csv
from itertools import zip_longest


# load all gathered csv data TODO:
X_train, y_train = [], []


print("Train Data Shape", len(X_train), len(X_train[0]))
print("Target Data Shape", len(y_train))


# define best founded parameters for random forest TODO:
best_params = {
    'min_samples_split': 1,
    'n_estimators': 1,
    'max_features': 1
}
# train Random Forrest regression with all gathered data
# and best founded parameters in previous random search
rf = RandomForestRegressor(min_samples_split=best_params['min_samples_split'],
                           n_estimators=best_params['n_estimators'],
                           max_features=best_params['max_features'])
rf.fit(X_train, y_train)


def random_forest_loss_predict(params):
    """
    Predict loss from trained Random Forest as surrogate benchmark model
    :param params:
    :return:
    """
    X_params = [params['lrate'], params['l2_reg'], params['n_epochs'] ]
    loss = rf.predict(X_params)
    return loss


# define space for TPE HPO
space = {"lrate": hp.uniform("lrate", 0, 1),
         "l2_reg": hp.uniform("l2_reg", 0, 1),
         # "batchsize": scope.int(hp.quniform("batchsize", 20, 2000, 1)),
         "n_epochs": scope.int(hp.quniform("n_epochs", 5, 2000, 1))}


def obj_func(params):
    """
    Objective function for TPE HPO
    :param params:
    :return: predicted loss from surrogate benchmark (random forest)
    """
    loss = random_forest_loss_predict(params)

    return {'loss': loss, 'status': STATUS_OK}


if __name__ == "__main__":
    num_repeat = 10

    for i in range(num_repeat):
        print(f'Run {i}/{num_repeat}')
        # perform TPE optimization and do logging
        trials = Trials()
        best_params = fmin(fn=obj_func,
                        space=space,
                        algo=tpe.suggest,
                        max_evals=100,
                        trials=trials)

        print("Best parameters:", best_params)
        print(trials.best_trial['result']['loss'])

        loss = trials.losses()
        val = trials.vals
        val['loss'] = loss
        print(val)

        # with open('best.json', 'w') as f:
        #     f.write(json.dumps({"Loss": trials.best_trial['result']['loss'],
        #                         "Best params": best_params}))

        filename = 'csv_data/SB_tpe{}.csv'.format(i)
        # header = ['lrate', 'l2_reg', 'batchsize', 'n_epochs', 'loss']
        header = ['lrate', 'l2_reg', 'n_epochs', 'loss']
        values = (val.get(key, []) for key in header)
        data = (dict(zip(header, row)) for row in zip(*values))
        with open(filename, 'w') as f:
            wrtr = csv.DictWriter(f, fieldnames=header)
            wrtr.writeheader()
            wrtr.writerows(data)



