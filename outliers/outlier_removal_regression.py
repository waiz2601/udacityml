#!/usr/bin/python3

import random
import numpy
import matplotlib.pyplot as plt
import joblib
from outlier_cleaner import outlierCleaner

### load up some practice data with outliers in it
ages = joblib.load(open("./practice_outliers_ages.pkl", "rb"))
net_worths = joblib.load(open("./practice_outliers_net_worths.pkl", "rb"))

### ages and net_worths need to be reshaped into 2D numpy arrays
ages = numpy.reshape(numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape(numpy.array(net_worths), (len(net_worths), 1))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as score

ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)

### Fit the initial regression model
reg = LinearRegression()
reg.fit(ages_train, net_worths_train)
print("Initial Slope:", reg.coef_)
print("Initial Intercept:", reg.intercept_)
print("Initial R2 Score:", score(net_worths_test, reg.predict(ages_test)))

### Plot the initial regression line
try:
    plt.plot(ages, reg.predict(ages), color="blue")
except NameError:
    pass
plt.scatter(ages, net_worths)
plt.xlabel("ages")
plt.ylabel("net worths")
plt.show()

### Identify and remove the most outlier-y points
cleaned_data = []
try:
    predictions = reg.predict(ages_train)
    cleaned_data = outlierCleaner(predictions, ages_train, net_worths_train)
except NameError:
    print("Your regression object doesn't exist, or isn't named reg")
    print("Can't make predictions to use in identifying outliers")

### Only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    ages, net_worths, errors = zip(*cleaned_data)
    ages = numpy.reshape(numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape(numpy.array(net_worths), (len(net_worths), 1))

    ### Refit your cleaned data
    try:
        reg.fit(ages, net_worths)
        plt.plot(ages, reg.predict(ages), color="blue")
        print("New Slope:", reg.coef_)
        print("New Intercept:", reg.intercept_)
        print("New R2 Score:", score(net_worths_test, reg.predict(ages_test)))
    except NameError:
        print("You don't seem to have regression imported/created,")
        print("   or else your regression object isn't named reg")
        print("   either way, only draw the scatter plot of the cleaned data")
    plt.scatter(ages, net_worths)
    plt.xlabel("ages")
    plt.ylabel("net worths")
    plt.show()
else:
    print("outlierCleaner() is returning an empty list, no refitting to be done")


