
import csv 


header = ['lrate', 'l2_reg','n_epochs','loss']

data = []

#for each of the 100 evaluations
for i in range(1,6):

    with open('/Users/mrsalwer/Downloads/Spearmint-master/examples/simple/output/0000000'+ str(i) + '.out') as f:
        lines = f.readlines()


    result = lines[-1:][0] #loss
    var3 = lines[-2:][0] #n_epochs
    var2 = lines[-3:][0] #l2_reg
    var1 = lines[-4:][0] #lrate

    data.append([var1[:-2],var2[:-2],var3[-2],result[-2]])


#create csv file
with open('/Users/mrsalwer/Downloads/Spearmint-master/examples/simple/run.csv', 'w') as c:

    writer = csv.writer(c)
    writer.writerow(header)
    for row in data:
        print(row)
        writer.writerow(row)

