from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import *
from collections import Counter
from sklearn import cross_validation, metrics
from sklearn.cross_validation import train_test_split

# fill the NaN-values in the given column with the non-NaN-median
def fillMedian(df, col, medianGiven = None):
    if medianGiven is None:
        median = df[col].median()
    else:
        median = medianGiven

    df[col] = df[col].fillna(median)

    return median

# fill the NaN-values in the given column with the non-NaN-median matrix 
# separated by (sex, pclass)
def fillMedianClassSex(df, col, mediansGiven = None):
    if mediansGiven is None:
        medians = np.zeros((2,3))
        for i in range(0, 2):
            for j in range(0, 3):
                medians[i,j] = df[(df.Sex == i) & (df.Pclass == j+1)][col].dropna().median()
    else:
        medians = mediansGiven

    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[ (df[col].isnull()) & (df.Sex == i) & (df.Pclass == j+1), col] = medians[i,j]

    return medians

# fill a column with given value
def fillWith(df, col, fillVal):
    assert( fillVal != None)
    df[col] = df[col].fillna(fillVal)

# fill the given column using the given method
def fillColumn(df, col, method, fillVal = None):
    if method == 0:
        return fillMedian(df, col, fillVal)
    elif method == 1:
        return fillMedianClassSex(df, col, fillVal)
    elif method == 2:
        assert(fillVal != None)
        return fillWith(df, col, fillVal)
    else:
        assert(False)

def convertToOneHot(df, col):
    pass

def convertToInt(df, col, dctGiven = None):
    if dctGiven is None:
        dct = {x:k for k,x in enumerate(df[col].unique())}
    else:
        dct = dctGiven

    df[col] = df[col].map(dct)

    return dct

# read stuff
dft = pd.read_csv('test.csv', header=0)
df = pd.read_csv('train.csv', header=0)


"""
Data enchrichment phase
"""

"""
dfage = df[['Pclass', 'SibSp', 'Parch', 'Sex', 'Age', 'Fare']].dropna()
clf_age = DecisionTreeClassifier(max_depth=5)
clf_age.fit(dfage[['Pclass', 'SibSp', 'Parch', 'Sex', 'Fare']].values.astype(float), dfage['Age'].astype(float))
clf_age.predict(
"""

# fill Age and Fare NaN-values
mediansAge = fillColumn(df, 'Age', 0)
fillColumn(dft, 'Age', 0, mediansAge)
mediansFare = fillColumn(df, 'Fare', 0)
fillColumn(dft, 'Fare', 0, mediansFare)


# fill the missing embarked values with the most-occuring value
fillColumn(df, 'Embarked', 2, sorted(Counter(df['Embarked'].values).items(), key=lambda x: x[1])[-1][0])

# convert some stuff
dctSex = convertToInt(df, 'Sex')
convertToInt(dft, 'Sex', dctSex)
dctEmb = convertToInt(df, 'Embarked')
convertToInt(dft, 'Embarked', dctEmb)

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
dft['FamilySize'] = dft['SibSp'] + dft['Parch'] + 1

# choose which columns to predict with
predictors = ['Pclass', 'SibSp', 'Parch', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']


""" 
Training phase 
"""

classifiers = [
    [RandomForestClassifier(), 
     {"n_estimators": [50, 100, 200, 500], "max_depth": [2, 3, 5, 8]}], 
    [GradientBoostingClassifier(), 
     {"n_estimators": [50, 100, 200, 500], "max_depth": [2, 3, 5, 8]}]
]
    #forest = GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)

nrepeat = 4
results = np.zeros([len(classifiers), nrepeat])
estimators = np.zeros([len(classifiers), nrepeat], dtype=object)

for rs in range(nrepeat):
    traindata = df[predictors].values.astype(float)
    targets = df.Survived.astype(float)

    Xtr, Xte, ytr, yte = train_test_split( traindata, targets, test_size = 1/3, random_state=rs)

    # train classifier
    for i, [clf, grid] in enumerate(classifiers):
        gs = GridSearchCV(clf, grid, cv=5, n_jobs=-1)
        gs.fit(Xtr, ytr)

        results[i, rs] = gs.score(Xte, yte)
        estimators[i, rs] = gs.best_estimator_
        print results

best_classifier = np.argmax(np.mean(results, axis=1))
best_model = estimators[best_classifier, np.argmax(results, axis=1)[best_classifier]]
print best_model



"""
Testing phase
"""

best_model.fit(traindata, targets)

# predict final output
output = best_model.predict_proba(dft[predictors].astype(float))[:,1]
output = (output > 0.5).astype(int)

submission = pd.DataFrame({ 'PassengerId': dft['PassengerId'], 'Survived': output });

submission.to_csv('model_%f.csv' % np.max(np.mean(results, axis=1)), index=False)
