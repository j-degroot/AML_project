from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler, train_test_split
from csv import reader
import numpy as np
import scipy.stats as stats
import pickle
import os

print(os.cpu_count())

data_X = []
data_y = []
opts = ['random', 'tpe', 'run']
for op in opts:
    for i in range(10):
        with open('Data-NoSMAC/' + op + str(i) +'.csv', 'r') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)
            # Check file as empty
            if header != None:
                # Iterate over each row after the header in the csv
                for row in csv_reader:
                    data_X.append([float(i) for i in row[:3]])
                    data_y.append(float(row[3]))

# HPO on RF Surrogate

# Shuffle data
c = list(zip(data_X, data_y))
np.random.shuffle(c)
data_X, data_y = zip(*c)
data_X = np.array(data_X)

print('data loaded')
#Split data



split_point = int(len(data_y)/2)
X_train, y_train, X_test, y_test = train_test_split(data_X, data_y, test_size=0.5, random_state=42)

param_distributions = dict(max_depth=stats.randint(3,18),
                           gamma=stats.uniform(1,9),
                           learning_rate=stats.loguniform(0.005, 0.5),
                           min_child_weight=stats.randint(1, 10))

xgb = XGBRegressor()
best_acc = 0
best_config = dict()
for config in ParameterSampler(param_distributions, n_iter=100, random_state=0):
    model = XGBRegressor(**config)
    XGBRegressor.fit(X_train, y_train)
    score = XGBRegressor.score(X_test, y_test)
    if score > best_acc:
        best_acc = score
        best_config = config

model = XGBClassifier(max_depth = search.best_params_['max_depth'], gamma = search.best_params_['gamma'], learning_rate = search.best_params_['learning_rate'], min_child_weight = search.best_params_['min_child_weight'])

model.fit(data_X, data_y) # fits on all data

model.save_model('XGB-SMAC.model')