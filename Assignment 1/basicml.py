from __future__ import division
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from explore import *
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

#estiamte ML on the other 3 courses?
featurematrix = np.array([stp, irp, dbp]).transpose()

#haal rijen met missing values er uit
targetcol = mlp
data = []
targets = []
for i, row in enumerate(featurematrix):
    if not any(v is None for v in row) and targetcol[i] is not None:
        data.append(row)
        targets.append(targetcol[i])

nsamples = len(targets)
Xtr, Xte, ytr, yte = train_test_split( data, targets, test_size = 1/3, random_state=42)


classifiers = [
    DecisionTreeClassifier(max_depth=5),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB() ]

for clf in classifiers:
    print clf
    clf.fit(Xtr, ytr)
    print clf.score(Xte, yte)

#use decision tree and SVM with crossvalidation to find better results

clf = GridSearchCV(DecisionTreeClassifier(), {"max_depth": range(1, 15)}, cv=5)
clf.fit(Xtr, ytr)
print clf.best_params_
print clf.score(Xte, yte)

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
clf = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5)
clf.fit(Xtr, ytr)
print clf.best_params_
print clf.score(Xte, yte)
