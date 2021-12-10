# convert JSON output data to csv

import json

'/Users/Jurren/Documents/GitHub/AML_project/smac/smac3-output_data'
import os
os.getcwd()


for i in range(1,11):
    # Opening JSON file
    json_file = f'/optimizers/sma'

    f = open('data.json')

# returns JSON object as
# a dictionary
data = json.load(f)

# Iterating through the json
# list
for i in data['emp_details']:
    print(i)

# Closing file
f.close()
