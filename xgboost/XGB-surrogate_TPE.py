from xgboost import XGBRegressor
from sklearn.model_selection import ParameterSampler, train_test_split
from csv import reader
import numpy as np
import scipy.stats as stats

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

# Shuffle data
c = list(zip(data_X, data_y))
np.random.shuffle(c)
data_X, data_y = zip(*c)
data_X = np.array(data_X)

print('Data Loaded')

X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.5, random_state=41)
print(type(X_train))
print(type(y_train))

param_distributions = dict(max_depth=stats.randint(3,18),
                           gamma=stats.uniform(1,9),
                           learning_rate=stats.loguniform(0.005, 0.5),
                           min_child_weight=stats.randint(1, 10))

xgb = XGBRegressor()
best_acc = 0
best_config = dict()
for config in ParameterSampler(param_distributions, n_iter=100, random_state=0):
    # import pdb; pdb.set_trace()
    model = XGBRegressor(**config)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    if score > best_acc:
        best_acc = score
        best_config = config

model = XGBRegressor(**best_config)

model.fit(data_X, data_y) # fits on all data

model.save_model('XGB-TPE.model')