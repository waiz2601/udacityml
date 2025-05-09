#!/usr/bin/python3

""" 
    Skeleton code for k-means clustering mini-project.
"""

import os
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)

# Find the maximum and minimum values of the "exercised_stock_options" feature
exercised_stock_options = [person["exercised_stock_options"] for person in data_dict.values() if person["exercised_stock_options"] != "NaN"]

max_exercised_stock_options = max(exercised_stock_options)
min_exercised_stock_options = min(exercised_stock_options)

print("Maximum value of exercised_stock_options:", max_exercised_stock_options)
print("Minimum value of exercised_stock_options:", min_exercised_stock_options)
print("stock",(1000000-min_exercised_stock_options)/(max_exercised_stock_options-min_exercised_stock_options))


salary_list = [person["salary"] for person in data_dict.values() if person["salary"] != "NaN"]

if salary_list:  # Ensure the list is not empty before applying min/max
    max_salary = max(salary_list)
    min_salary = min(salary_list)

    print("Maximum value of salary:", max_salary)
    print("Minimum value of salary:", min_salary)
    print('salary',(200000-min_salary)/(max_salary-min_salary))
else:
    print("No valid salary values found in the dataset.")


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2, feature_3]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )

# Apply feature scaling
scaler = MinMaxScaler()
finance_features = scaler.fit_transform(finance_features)

### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
for f1, f2, _ in finance_features:
    plt.scatter( f1, f2 )
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(finance_features)
pred = kmeans.predict(finance_features)

### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters_3_features_scaled.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print("No predictions object named pred found, no clusters to plot")
