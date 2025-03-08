#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC # Support Vector Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
grid={
    'kernel':['rbf','linear'],
    'C':[10.0,100.0,1000.0,10000.0]
}




clf = SVC()
grid_search = GridSearchCV(clf, param_grid=grid, scoring='accuracy',cv=5)
grid_search.fit(features_train,labels_train)
t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")
t1=time()
pred=clf.predict(features_test)
print("Prediction Time:", round(time()-t1, 3), "s")
accuracy = accuracy_score(labels_test, pred)
print("Accuracy:", accuracy)
print(grid_search.best_params_)
print("Prediction for element 10:", pred[10])
print("Prediction for element 26:", pred[26])
print("Prediction for element 50:", pred[50])
num_chris=sum(pred==1)
print("Number of predictions that are 1:", num_chris)


#########################################################
### your code goes here ###


#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''


#########################################################
