from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from csv import reader
import numpy as np
import scipy.stats as stats 
import pickle
from sklearn.preprocessing import scale, normalize, MinMaxScaler

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

data_X = []
data_y = []
opts = ['random','smac','run']
for op in opts: 
    for i in range(10):
        with open('All_Data/' + op + str(i) +'.csv', 'r') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)
            # Check file as empty
            if header != None:
                # Iterate over each row after the header in the csv
                for row in csv_reader:
                    data_X.append([float(i) for i in row[:3]])
                    data_y.append(float(row[3]))

# HPO on RF Surrogate

#Split data
split_point = int(len(data_y)/2)

# training_X = data_X[:split_point]
# training_y = data_y[:split_point]
# testing_X = data_X[split_point:]
# testing_y = data_y[split_point:]



regr = GradientBoostingRegressor()

# data_y = NormalizeData(data_y)
# data_X = normalize(data_X,axis=0)


distributions = dict(max_depth=stats.randint(2,25), max_features=stats.uniform(loc = 0.1, scale=0.95), min_samples_leaf = stats.uniform(loc = 0.1, scale=0.95))


clf = RandomizedSearchCV(regr, distributions, n_iter = 100)

search = clf.fit(data_X,data_y)

print(search.best_params_)

model = GradientBoostingRegressor( max_depth = search.best_params_['max_depth'], max_features = search.best_params_['max_features'],  min_samples_leaf = search.best_params_['min_samples_leaf'])

model.fit(data_X,data_y)

filename = 'GB-TPE.sav'

print(model.predict([[0.9,0.9,100]]))

print(model.predict([[0.6,0.1,2]]))

pickle.dump(model, open(filename, 'wb'))





