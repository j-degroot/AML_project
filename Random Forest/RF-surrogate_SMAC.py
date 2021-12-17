from sklearn.ensemble import RandomForestRegressor
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
# c = list(zip(data_X, data_y)) 
# np.random.shuffle(c)
# data_X, data_y = zip(*c)

#Split data
split_point = int(len(data_y)/2)

# training_X = data_X[:split_point]
# training_y = data_y[:split_point]
# testing_X = data_X[split_point:]
# testing_y = data_y[split_point:]



regr = RandomForestRegressor()


distributions = dict(n_estimators=stats.randint(20, 200), min_samples_split=stats.uniform(loc = 0.01, scale=1), max_features=stats.uniform(loc = 0.1, scale=1))



clf = RandomizedSearchCV(regr, distributions, n_iter = 100, random_state=0)

search = clf.fit(data_X, data_y)

model = RandomForestRegressor(max_features = search.best_params_['max_features'], min_samples_split = search.best_params_['min_samples_split'], n_estimators = search.best_params_['n_estimators'])

model.fit(data_X, data_y)

filename = 'RF-SMAC.sav'

pickle.dump(model, open(filename, 'wb'))





