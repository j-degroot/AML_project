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

from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario

import time


filename = 'RF-SMAC.sav'
rf_surrogate = pickle.load(open(filename, 'rb'))


print('Data importing and splitting complete')

def random_forest_loss_predict(config):
    """
    Predict loss from trained Random Forest as surrogate benchmark model
    :param params:
    :return:
    """
    X_params = np.array([config['lrate'], config['l2_reg'], config['n_epochs'] ]).reshape(1, -1)
    loss = rf_surrogate.predict(X_params)
    print(X_params)

    return float(loss)

configspace = ConfigurationSpace()
configspace.add_hyperparameter(UniformFloatHyperparameter("lrate", 0, 1))
configspace.add_hyperparameter(UniformFloatHyperparameter("l2_reg", 0, 1))
configspace.add_hyperparameter(UniformIntegerHyperparameter("n_epochs", 5, 2000))

if __name__ == "__main__":
    # Provide meta data for the optimization
    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality (alternatively runtime)
        "runcount-limit": 100,  # Max number of function evaluations (the more the better)
        "cs": configspace,
        "abort_on_first_run_crash": False,
        "deterministic" : True
    })
    for seed in range(0, 10):
        smac = SMAC4BB(scenario=scenario, tae_runner=random_forest_loss_predict)
        best_found_config = smac.optimize()
        print(best_found_config)