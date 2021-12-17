#for every run, make list of best loss found so far. 

#Plot mean of lists and s.d.

from csv import reader
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe

loss_lists_tpe = dict()
loss_lists_smac = dict()
loss_lists_random = dict()


for i in range(10):
    losses_tpe = []
    losses_smac = []
    losses_random = []
    with open('tpe-gb-surrogate-runs/gb_tpe'  + str(i) +'.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Check file as empty
        if header != None:
            # Iterate over each row after the header in the csv
            for row in csv_reader:
                losses_tpe.append(float(row[3]))
    with open('smac-gb-surrogate-runs/gb_smac'  + str(i) +'.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Check file as empty
        if header != None:
            # Iterate over each row after the header in the csv
            for row in csv_reader:
                losses_smac.append(float(row[3]))
    with open('random-gb-surrogate-runs/gb_'  + str(i) +'.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Check file as empty
        if header != None:
            # Iterate over each row after the header in the csv
            for row in csv_reader:
                print(row)
                losses_random.append(float(row[3][1:-1]))
    
    
    loss_lists_tpe[str(i)] = losses_tpe
    loss_lists_smac[str(i)] = losses_smac
    loss_lists_random[str(i)] = losses_random



only_losses_tpe = []
only_losses_smac = []
only_losses_random = []
for i in range(10):
    losses_smac = loss_lists_smac[str(i)]
    losses_tpe = loss_lists_tpe[str(i)]
    losses_random = loss_lists_random[str(i)]
    for z in range(len(losses_tpe)):
        if z > 0:  
            losses_tpe[z] = min(losses_tpe[:z+1])
            losses_smac[z] = min(losses_smac[:z+1])
            losses_random[z] = min(losses_random[:z+1])

    only_losses_tpe.append(losses_tpe)
    only_losses_smac.append(losses_smac)
    only_losses_random.append(losses_random)


def column(matrix, i):

    return [row[i] for row in matrix]


x_tpe = np.array(range(0,100))
y_tpe = []
st_devs_tpe = []

x_smac = np.array(range(0,100))
y_smac = []
st_devs_smac = []

x_random = np.array(range(0,100))
y_random = []
st_devs_random = []





for i in x_tpe: 
    ten_runs_tpe = column(only_losses_tpe, i)
    y_tpe.append(sum(ten_runs_tpe)/len(ten_runs_tpe))
    st_devs_tpe.append(np.std(ten_runs_tpe))

    ten_runs_smac = column(only_losses_smac, i)
    y_smac.append(sum(ten_runs_smac)/len(ten_runs_smac))
    st_devs_smac.append(np.std(ten_runs_smac))

    ten_runs_random = column(only_losses_random, i)
    y_random.append(sum(ten_runs_random)/len(ten_runs_random))
    st_devs_random.append(np.std(ten_runs_random))


print(st_devs_tpe)
print(st_devs_random)



fig, ax = plt.subplots()

y_tpe = np.array(y_tpe)
st_devs_tpe = np.array(st_devs_tpe)

y_smac = np.array(y_smac)
st_devs_smac = np.array(st_devs_smac)

y_random = np.array(y_random)
st_devs_random = np.array(st_devs_random)


ax.plot(x_smac, y_smac, color = '#e76100', linestyle = 'solid', lw = 2.5, label= 'SMAC GRADIENT BOOSTING', path_effects=[pe.Stroke(linewidth=3.5, foreground='w'), pe.Normal()])


ax.plot(x_tpe, y_tpe, 'k--', lw = 2.3, label= 'TPE GRADIENT BOOSTING', path_effects=[pe.Stroke(linewidth=3.5, foreground='w'), pe.Normal()])

ax.plot(x_random, y_random, color = '#5D3C99', linestyle =  (0, (3, 1, 1, 1)), lw = 2.5, label= 'RANDOM SEARCH GRADIENT BOOSTING', path_effects=[pe.Stroke(linewidth=3.5, foreground='w'), pe.Normal()])



ax.set_ylim([0.1, 0.18])
ax.set_xlabel('# Evaluations')
ax.set_xscale('log')
ax.set_ylabel('Best Loss Found So Far')
ax.set_title('Optimizers on Sklearn Gradient Boosting Surrogate')
ax.grid()
ax.fill_between(x_smac, y_smac -st_devs_smac, y_smac+st_devs_smac, color = '#e76100', alpha= 0.5)
ax.fill_between(x_tpe, y_tpe -st_devs_tpe, y_tpe+st_devs_tpe, color = 'k', alpha= 0.3)

ax.fill_between(x_random, y_random -st_devs_random, y_random+st_devs_random, color = '#5D3C99', alpha= 0.5)

ax.legend()

plt.show()