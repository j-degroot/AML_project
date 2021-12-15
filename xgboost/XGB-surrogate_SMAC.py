from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from csv import reader
import numpy as np
import scipy.stats as stats
import pickle

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

#Split data
split_point = int(len(data_y)/2)

param_distributions = dict(max_depth=stats.randint(3,18),
                           gamma=stats.uniform(1,9),
                           learning_rate=stats.loguniform(0.005, 0.5),
                           min_child_weight=stats.randint(1, 10))

xgb = XGBRegressor()

clf = RandomizedSearchCV(xgb, param_distributions, cv=2, n_iter=100, random_state=0, n_jobs=-1)

search = clf.fit(data_X, data_y)

model = XGBClassifier(max_features = search.best_params_['max_features'], min_samples_split = search.best_params_['min_samples_split'], n_estimators = search.best_params_['n_estimators'])

model.fit(data_X, data_y) # fits on all data

filename = 'XGB-SMAC.sav'

pickle.dump(model, open(filename, 'wb'))