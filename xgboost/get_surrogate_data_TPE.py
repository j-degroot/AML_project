from hyperopt.pyll import scope
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import numpy as np
import csv
import xgboost as xgb

# define space for TPE HPO
space = {"lrate": hp.uniform("lrate", 0, 1),
         "l2_reg": hp.uniform("l2_reg", 0, 1),
         # "batchsize": scope.int(hp.quniform("batchsize", 20, 2000, 1)),
         "n_epochs": scope.int(hp.quniform("n_epochs", 5, 2000, 1))}

surrogate_model = xgb.Booster()
surrogate_model.load_model('xgboost/XGB-TPE.txt')

def surrogate(config):
    global surrogate_model
    x = np.array([[config['lrate'], config['l2_reg'], config['n_epochs']]])
    x = xgb.DMatrix(x)
    return float(surrogate_model.predict(x))  # SMAC minimizes the objective functio


if __name__ == "__main__":
    num_repeat = 2

    for i in range(num_repeat):
        print(f'Run {i}/{num_repeat-1}')
        # perform TPE optimization and do logging
        trials = Trials()
        best_params = fmin(fn=surrogate,
                        space=space,
                        algo=tpe.suggest,
                        max_evals=5,
                        trials=trials)

        print("Best parameters:", best_params)
        print(trials.best_trial['result']['loss'])

        loss = trials.losses()
        val = trials.vals
        val['loss'] = loss
        print(val)

        filename = 'TPE_xgb_hpo{}.csv'.format(i)
        # header = ['lrate', 'l2_reg', 'batchsize', 'n_epochs', 'loss']
        header = ['lrate', 'l2_reg', 'n_epochs', 'predicted_loss']
        values = (val.get(key, []) for key in header)
        data = (dict(zip(header, row)) for row in zip(*values))
        with open(filename, 'w') as f:
            wrtr = csv.DictWriter(f, fieldnames=header)
            wrtr.writeheader()
            wrtr.writerows(data)