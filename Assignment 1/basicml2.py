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
scores_1 = [0]*10
scores_2 = [0]*10
for i in range(10):
    Xtr, Xte, ytr, yte = train_test_split( data, targets, test_size = 1/3, random_state=i)


    clf = GridSearchCV(DecisionTreeClassifier(), {"max_depth": range(1, 15)}, cv=5)
    clf.fit(Xtr, ytr)
    print clf.best_params_
    scores_1[i] = clf.score(Xte, yte)

    param_grid = {'C': [1e1, 1e2, 1e3, 1e4, 1e5], 'gamma': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
    clf = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5)
    clf.fit(Xtr, ytr)
    print clf.best_params_
    scores_2[i] = clf.score(Xte, yte)

print sum(scores_1)/10, sum(scores_2)/10
