#!/usr/bin/python3


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from time import time

    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess



features_train, features_test, labels_train, labels_test = preprocess()



clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")


t0 = time()
pred = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

accuracy = accuracy_score(labels_test, pred)
print("Accuracy:", accuracy)



