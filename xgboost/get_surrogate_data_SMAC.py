import os

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario
from csv import reader
import xgboost as xgb
import numpy as np
import time
import pickle

configspace = ConfigurationSpace()
configspace.add_hyperparameter(UniformFloatHyperparameter("lrate", 0, 10))
configspace.add_hyperparameter(UniformFloatHyperparameter("l2_reg", 0, 1))
configspace.add_hyperparameter(UniformIntegerHyperparameter("n_epochs", 5, 2000))


def surrogate(config):
    global surrogate_model
    x = np.array([[config['lrate'], config['l2_reg'], config['n_epochs']]])
    return surrogate_model.predict(x)[0]  # SMAC minimizes the objective function

# Change data_home to wherever to where you want to download your data

if __name__ == "__main__":
    print(os.getcwd())
    surrogate_model = xgb.Booster()
    surrogate_model.load_model('xgboost/XGB-SMAC.sav')
    # Provide meta data for the optimization
    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality (alternatively runtime)
        "runcount-limit": 100,  # Max number of function evaluations (the more the better)
        "cs": configspace,
        'deterministic': True,
        "abort_on_first_run_crash": False
    })
    for seed in range(0, 10):
        smac = SMAC4BB(scenario=scenario, tae_runner=surrogate, rng=seed)
        best_found_config = smac.optimize()
        print(best_found_config)