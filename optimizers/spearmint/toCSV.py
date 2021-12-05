
import csv 


header = ['lrate', 'l2_reg','n_epochs','loss']

data = []

#for each of the 100 evaluations
for i in range(1,50): 
    if len(str(i)) == 1:
        with open('/Users/mrsalwer/Downloads/Spearmint-master/examples/simple/output/0000000'+ str(i) + '.out') as f:
            lines = f.readlines()
    elif len(str(i)) == 2:
        with open('/Users/mrsalwer/Downloads/Spearmint-master/examples/simple/output/000000'+ str(i) + '.out') as f:
            lines = f.readlines()
    elif len(str(i)) == 3:
        with open('/Users/mrsalwer/Downloads/Spearmint-master/examples/simple/output/00000'+ str(i) + '.out') as f:
            lines = f.readlines()


    result = lines[-1:][0] #loss
    var3 = lines[-2:][0] #n_epochs
    var2 = lines[-3:][0] #l2_reg
    print(var2)
    var1 = lines[-4:][0] #lrate


    data.append([var1,var2,var3,result])


#create csv file
with open('/Users/mrsalwer/Downloads/Spearmint-master/examples/simple/run.csv', 'w') as c:

    writer = csv.writer(c)
    writer.writerow(header)
    for row in data:
        writer.writerow(row)

