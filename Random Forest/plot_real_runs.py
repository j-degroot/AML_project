#for every run, make list of best loss found so far. 

#Plot mean of lists and s.d.

from csv import reader
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe

loss_lists_tpe = dict()
loss_lists_smac = dict()

for i in range(10):
    losses_tpe = []
    losses_smac = []
    with open('tpe-real-runs/tpe'  + str(i) +'.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Check file as empty
        if header != None:
            # Iterate over each row after the header in the csv
            for row in csv_reader:
                losses_tpe.append(float(row[3]))
    with open('smac-real-runs/smac'  + str(i) +'.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Check file as empty
        if header != None:
            # Iterate over each row after the header in the csv
            for row in csv_reader:
                losses_smac.append(float(row[3]))
    
    loss_lists_tpe[str(i)] = losses_tpe
    loss_lists_smac[str(i)] = losses_smac



only_losses_tpe = []
only_losses_smac = []
for i in range(10):
    losses_smac = loss_lists_smac[str(i)]
    losses_tpe = loss_lists_tpe[str(i)]
    for z in range(len(losses_tpe)):
        if z > 0:  
            losses_tpe[z] = min(losses_tpe[:z+1])
            losses_smac[z] = min(losses_smac[:z+1])

    only_losses_tpe.append(losses_tpe)
    only_losses_smac.append(losses_smac)


def column(matrix, i):

    return [row[i] for row in matrix]


x_tpe = np.array(range(0,100))
y_tpe = []
st_devs_tpe = []

x_smac = np.array(range(0,100))
y_smac = []
st_devs_smac = []




for i in x_tpe: 
    ten_runs_tpe = column(only_losses_tpe, i)
    y_tpe.append(sum(ten_runs_tpe)/len(ten_runs_tpe))
    st_devs_tpe.append(np.std(ten_runs_tpe))

    ten_runs_smac = column(only_losses_smac, i)
    y_smac.append(sum(ten_runs_smac)/len(ten_runs_smac))
    st_devs_smac.append(np.std(ten_runs_smac))


print(len(st_devs_tpe))



fig, ax = plt.subplots()
y_tpe = np.array(y_tpe)
st_devs_tpe = np.array(st_devs_tpe)

y_smac = np.array(y_smac)
st_devs_smac = np.array(st_devs_smac)


ax.plot(x_smac, y_smac, color = '#e76100', linestyle = 'solid', lw = 2.5, label= 'SMAC REAL', path_effects=[pe.Stroke(linewidth=3.5, foreground='w'), pe.Normal()])

ax.plot(x_tpe, y_tpe, 'k--', lw = 2.3, label= 'TPE REAL', path_effects=[pe.Stroke(linewidth=3, foreground='w'), pe.Normal()])



ax.set_ylim([0.1, 0.18])
ax.set_xlabel('# Evaluations')
ax.set_xscale('log')
ax.set_ylabel('Best Loss Found So Far')
ax.set_title('Optimizers on Real Logistic Regression MNIST')
ax.grid()
ax.fill_between(x_smac, y_smac -st_devs_smac, y_smac+st_devs_smac, color = '#e76100', alpha= 0.5)
ax.fill_between(x_tpe, y_tpe -st_devs_tpe, y_tpe+st_devs_tpe, color = 'k', alpha= 0.3)

ax.legend()

plt.show()