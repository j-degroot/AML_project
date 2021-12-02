import numpy as np
import math
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml



def branin(lrate, l2_reg,n_epochs):

    mnist = fetch_openml('mnist_784')

    x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.15, random_state=0)


    clf = SGDClassifier(loss = 'log', learning_rate='constant', eta0=lrate, penalty='elasticnet', l1_ratio=l2_reg, max_iter=n_epochs, shuffle=True)

    clf.fit(x_train, y_train)

    result = clf.score(x_test, y_test)

    print result
    #time.sleep(np.random.randint(60))
    return result

# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params['lrate'][0]
    print params['l2_reg'][0]
    print params['n_epochs'][0]
    return branin(params['lrate'], params['l2_reg'],params['n_epochs'])

