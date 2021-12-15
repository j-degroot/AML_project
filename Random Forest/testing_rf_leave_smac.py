
from csv import reader
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe

from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr 
from matplotlib.markers import MarkerStyle
import pickle

data_X = []
data_y = []
opts = ['smac']
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

filename = 'RF-SMAC.sav'
rf_surrogate = pickle.load(open(filename, 'rb'))


y_actual = data_y
y_predicted = []



for i in data_X:
    pred = rf_surrogate.predict(np.array(i).reshape(1,-1))
    y_predicted.append(pred)


full_size = len(y_actual)
size = int(full_size/3)
y_actual, y_predicted = zip(*sorted(zip(y_actual, y_predicted)))

cc = spearmanr(y_actual, y_predicted)
rms = mean_squared_error(y_actual, y_predicted, squared=False)

print(cc, rms)



fig, ax = plt.subplots()

ax.plot([0, 1], [0, 1], transform=ax.transAxes, color = 'k',linestyle='dashed')

ax.axvline(x = y_actual[size], color = 'k', linestyle = 'dashed', alpha = 0.75, lw=1)
ax.axvline(x = y_actual[full_size-size], color = 'k',linestyle = 'dashed', alpha = 0.75, lw = 1)


ax.scatter(y_actual[:size], y_predicted[:size], s = 3, marker = MarkerStyle(marker='.', fillstyle='none'), color = 'g')
ax.scatter(y_actual[size:full_size-size], y_predicted[size:full_size-size], s = 3, marker = MarkerStyle(marker='.', fillstyle='none'), color = 'grey')
ax.scatter(y_actual[full_size-size:], y_predicted[full_size-size:], s = 3, marker = MarkerStyle(marker='.', fillstyle='none'), color = 'r')


ax.set_xlabel('True performance')
ax.set_xlim([0.09, 0.25])
ax.set_ylim([0.09, 0.25])
ax.set_ylabel('Model prediction')
ax.set_title('RF surrogaate predictions on leave-SMAC out data')
ax.grid()


plt.show()