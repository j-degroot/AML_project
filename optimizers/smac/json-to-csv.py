import json
import csv

for i in range(10):
    lines = []
    f = open('SMAC-DATA-RAW/run_'+str(i)+'/runhistory.json')
    data = json.load(f)
    

    for z in range(1,101):
        line = data['configs'][str(z)]
        loss = data['data'][z-1][1][0]
        lines.append([line['lrate'], line['l2_reg'], line['n_epochs'], loss])
        

    with open('SMAC-csv/' + 'smac' + str(i) +'.csv', 'w') as c:
        writer = csv.writer(c)
        writer.writerow(['lrate', 'l2_reg','n_epochs','loss'])
        for line in lines:
            writer.writerow(line)