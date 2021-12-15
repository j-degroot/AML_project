from mnist import MNIST
import numpy as np
import math
from sklearn.linear_model import SGDClassifier
import multiprocessing as mp
import os


def branin(lrate, l2_reg,n_epochs):

    mndata = MNIST('MNIST')

    x_train, y_train = mndata.load_training() 
    
    x_test, y_test = mndata.load_testing()

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    clf = SGDClassifier(loss = 'log', learning_rate='constant', eta0=lrate, penalty='elasticnet', l1_ratio=l2_reg, max_iter=n_epochs, shuffle=True, n_jobs=os.environ['SLURM_JOB_CPUS_PER_NODE'])

    clf.fit(x_train, y_train)

    result = clf.score(x_test, y_test)

    print (1 - result)
    #time.sleep(np.random.randint(60))
    return 1 - result

# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params['lrate'][0]
    print params['l2_reg'][0]
    print params['n_epochs'][0]
    return branin(params['lrate'], params['l2_reg'],params['n_epochs'])