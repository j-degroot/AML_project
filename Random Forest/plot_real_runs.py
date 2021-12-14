#for every run, make list of best loss found so far. 

#Plot mean of lists and s.d.

from csv import reader
import matplotlib.pyplot as plt
import numpy as np

loss_lists = dict()

for i in range(10):
    losses = []
    with open('tpe-real-runs/tpe'  + str(i) +'.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Check file as empty
        if header != None:
            # Iterate over each row after the header in the csv
            for row in csv_reader:
                losses.append(float(row[3]))
    
    loss_lists[str(i)] = losses


only_losses = []
for i in range(10):
    losses = loss_lists[str(i)]
    for z in range(len(losses)):
        if z > 0:  
            losses[z] = min(losses[:z+1])
    only_losses.append(losses)


def column(matrix, i):

    return [row[i] for row in matrix]


x = range(0,100)
y = []
st_devs = []

for i in x: 
    ten_runs = column(only_losses, i)
    y.append(sum(ten_runs)/len(ten_runs))
    st_devs.append(np.std(ten_runs))

print(len(st_devs))



fig, ax = plt.subplots()
x = np.array(x)
y = np.array(y)
st_devs = np.array(st_devs)
ax.plot(x, y, 'k--', lw = 1.4, label= 'TPE')
ax.set_ylim([0.1, 0.18])
ax.set_xlabel('# Evaluations')
ax.set_ylabel('Best Loss Found So Far')
ax.set_title('Optimizers on Real Logistic Regression MNIST')
ax.grid()

ax.fill_between(x, y -st_devs, y+st_devs, color = 'g', alpha= 0.3)
ax.legend()

plt.show()