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


filename = 'GB-TPE.sav'
rf_surrogate = pickle.load(open(filename, 'rb'))

def GB_loss_predict(params):
    """
    Predict loss from trained Random Forest as surrogate benchmark model
    :param params:
    :return:
    """
    X_params = np.array([params['lrate'], params['l2_reg'], params['n_epochs'] ]).reshape(1, -1)
    loss = rf_surrogate.predict(X_params)
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
    loss = GB_loss_predict(params)

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

        filename = 'tpe-gb-surrogate-runs/gb_tpe{}.csv'.format(i)
        # header = ['lrate', 'l2_reg', 'batchsize', 'n_epochs', 'loss']
        header = ['lrate', 'l2_reg', 'n_epochs', 'loss']
        values = (val.get(key, []) for key in header)
        data = (dict(zip(header, row)) for row in zip(*values))
        with open(filename, 'w') as f:
            wrtr = csv.DictWriter(f, fieldnames=header)
            wrtr.writeheader()
            wrtr.writerows(data)
