#run tpe using surrogate 10 times
#after every iteration, append a list with a list of the value progression of the run.
#[[run1...],[run2...],[run3...]...]
import pickle
from hyperopt.pyll import scope
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import pickle
import numpy as np
import csv
from itertools import zip_longest

import xgboost as xgb

from sklearn.model_selection import ParameterSampler
import scipy.stats as stats


filename = 'XGB-TPE.txt'
surrogate_model = xgb.Booster()
surrogate_model.load_model(filename)
xgb_surrogate = surrogate_model

def random_forest_loss_predict(params):
    """
    Predict loss from trained Random Forest as surrogate benchmark model
    :param params:
    :return:
    """
    X_params = np.array([params['lrate'], params['l2_reg'], params['n_epochs'] ]).reshape(1, -1)
    loss = xgb_surrogate.predict(X_params)
    return loss


# define space for TPE HPO
space = {"lrate": hp.uniform("lrate", 0, 1),
         "l2_reg": hp.uniform("l2_reg", 0, 1),
         # "batchsize": scope.int(hp.quniform("batchsize", 20, 2000, 1)),
         "n_epochs": scope.int(hp.quniform("n_epochs", 5, 2002, 1))}


def obj_func(params):
    """
    Objective function for TPE HPO
    :param params:
    :return: predicted loss from surrogate benchmark (random forest)
    """
    loss = random_forest_loss_predict(params)

    return {'loss': loss, 'status': STATUS_OK}

param_distributions = dict(lrate=stats.uniform(0.0000001,0.98),
                    l2_reg=stats.uniform(0.0000001,0.98),
                    n_epochs=stats.randint(2, 2000))
if __name__ == "__main__":
    num_repeat = 10

    for i in range(num_repeat):

        rows = []
        print(f'Run {i}/{num_repeat}')
        # perform TPE optimization and do logging
        filename = 'random-xgb-surrogate-runs/xgb_{}.csv'.format(i)
        header = ['lrate', 'l2_reg', 'n_epochs', 'loss']

        with open(filename, 'w') as f:
            wrtr = csv.DictWriter(f, fieldnames=header)
            wrtr.writeheader()

        for config in ParameterSampler(param_distributions, n_iter=100):
            params = [config['lrate'], config['lrate'], config['n_epochs']]
            prediction = xgb_surrogate.predict(xgb.DMatrix(np.array(params).reshape(1,-1)))
            row = []
            params.append(prediction)
            rows.append(params)
        
        with open(filename, 'w') as f:
            wrtr = csv.writer(f)
            wrtr.writerow(header)    
            wrtr.writerows(rows)
