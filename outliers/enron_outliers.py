#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot as plt 
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit

### Read in data dictionary, convert to numpy array
data_dict = joblib.load(open("../final_project/final_project_dataset.pkl", "rb"))

### Identify potential outliers by printing the highest values
keys_to_remove = [key for key, value in data_dict.items() if value["salary"] != 'NaN' and value["salary"] > 1e6]

# Remove the identified keys
data_dict.pop("TOTAL", 0)
### Remove the outlier "TOTAL"



features = ["salary", "bonus"]


data = featureFormat(data_dict, features)

### Extract salary and bonus from the data
salary, bonus = data[:, 0], data[:, 1]

### Create a scatterplot
plt.scatter(salary, bonus)
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

### your code below



