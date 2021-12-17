from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario
import numpy as np

import time


# Change data_home to wherever to where you want to download your data
mnist = fetch_openml("mnist_784")

train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)

print('Data importing and splitting complete')

def train_logistic_regression(config):
    """
    Trains a Logistic Regression on the given hyperparameters, defined by config, and returns the accuracy
    on the validation data.

    Input:
        config (Configuration): Configuration object derived from ConfigurationSpace.

    Return:
        cost (float): Performance measure on the validation data.
    """
    model = SGDClassifier(loss = 'log',
                          learning_rate= 'constant',
                          eta0= config['lrate'],
                          penalty='elasticnet',
                          l1_ratio=config['l2_reg'],
                          max_iter=config['n_epochs'],
                          shuffle=True
                          )

    model.fit(train_img, train_lbl)
    return 1 - model.score(test_img, test_lbl)  # SMAC minimizes the objective function


configspace = ConfigurationSpace()
configspace.add_hyperparameter(UniformFloatHyperparameter("lrate", 0, 10))
configspace.add_hyperparameter(UniformFloatHyperparameter("l2_reg", 0, 1))
configspace.add_hyperparameter(UniformIntegerHyperparameter("n_epochs", 5, 2000))

if __name__ == "__main__":

    # Provide meta data for the optimization
    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality (alternatively runtime)
        "runcount-limit": 100,  # Max number of function evaluations (the more the better)
        "cs": configspace,
        'deterministic' : True
        'abort_on_first_crash' : False
    })
    for seed in range(0, 11):
        smac = SMAC4BB(scenario=scenario, tae_runner=train_logistic_regression, rng=seed)
        best_found_config = smac.optimize()
        print(best_found_config)
