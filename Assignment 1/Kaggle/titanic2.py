# step into the future
from __future__ import division

# standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# some stuff we might need
from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint

#sklearn stuff
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn import cross_validation, metrics
from sklearn.cross_validation import train_test_split

# classifiers
from sklearn.ensemble import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC

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
dfc = pd.concat((df, dft), axis = 0, ignore_index = True)


"""
Data enchrichment phase
"""

# convert some stuff
dctSex = convertToInt(df, 'Sex')
convertToInt(dft, 'Sex', dctSex)
dctEmb = convertToInt(df, 'Embarked')
convertToInt(dft, 'Embarked', dctEmb)

"""
dfage = df[['Pclass', 'SibSp', 'Parch', 'Sex', 'Age', 'Fare']].dropna()
clf_age = DecisionTreeClassifier(max_depth=5)
clf_age.fit(dfage[['Pclass', 'SibSp', 'Parch', 'Sex', 'Fare']].values.astype(float), dfage['Age'].astype(float))
clf_age.predict(
"""

# fill Age and Fare NaN-values
mediansAge = fillColumn(df, 'Age', 1)
fillColumn(df, 'Age', 1, mediansAge)
fillColumn(dft, 'Age', 1, mediansAge)
mediansFare = fillColumn(df, 'Fare', 1)
fillColumn(df, 'Fare', 1, mediansFare)
fillColumn(dft, 'Fare', 1, mediansFare)

# add the Title column
titles = {
    "Mr": 0, 
    "Miss": 1,                 #unmarried woman
    "Mlle": 1,                 #french for Miss; unmarried
    "Mrs": 2,                  #married woman
    "Mme": 2,                  #married woman
    "Ms": 2,                   #female; we don't know if she's married
    "Master": 3, 
    "Dr": 4, 
    "Rev": 4, 
    "Col": 4,                  #male military title
    "Major": 4,                #male military title
    "Capt": 4,                 #male military title
    "Lady": 4,                 #female
    "the Countess": 4,         #female
    "Dona": 4,                 #spanish for Lady; female
    "Jonkheer": 4,             #dutch for Lord; male
    "Don": 4,                  #spanish for Lord; male
    "Sir": 4,                  #something honorary; male
}

df["Title"] = df.Name.apply(lambda x: titles[x.replace('.', ',').split(',')[1].strip()])
dft["Title"] = dft.Name.apply(lambda x: titles[x.replace('.', ',').split(',')[1].strip()])

# fill the missing embarked values with the most-occuring value
fillColumn(df, 'Embarked', 2, sorted(Counter(df['Embarked'].values).items(), key=lambda x: x[1])[-1][0])

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
dft['FamilySize'] = dft['SibSp'] + dft['Parch'] + 1

# choose which columns to predict with
predictors = ['Pclass', 'SibSp', 'Parch', 'Sex', 'Age', 'Fare', 'Title']


""" 
Training phase 
"""

classifiers = [
    [RandomForestClassifier(), 
     {"n_estimators": [100, 200, 500], "max_depth": [2, 3, 5, 8]}], 
    [GradientBoostingClassifier(), 
     {"n_estimators": [100, 200, 500], "max_depth": [2, 3, 5, 8]}],
    [DecisionTreeClassifier(), {"max_depth": [2, 3, 5, 8]}],
    [SVC(), 
     {'C': [1e-1, 1e1, 1e2], 'gamma': ['auto', 1e-3, 1e-2, 1e-1]}],
]

nrepeat = 1
results = np.zeros([len(classifiers), nrepeat])
estimators = np.zeros([len(classifiers), nrepeat], dtype=object)

for rs in range(nrepeat):
    traindata = df[predictors].values.astype(float)
    targets = df.Survived.astype(float)

    Xtr, Xte, ytr, yte = train_test_split( traindata, targets, test_size = 1/3)

    # train classifier
    for i, [clf, grid] in enumerate(classifiers):
        gs = GridSearchCV(clf, grid, cv=5, n_jobs=-1)
        gs.fit(Xtr, ytr)

        results[i, rs] = gs.score(Xte, yte)
        estimators[i, rs] = gs.best_estimator_
        print results

best_classifier = np.argmax(np.mean(results, axis=1))
best_model = estimators[best_classifier, np.argmax(results, axis=1)[best_classifier]]
print "estimated score: ", np.mean(results, axis=1)[best_classifier], " model: ", best_model


"""
Testing phase
"""

best_model.fit(traindata, targets)

# predict final output
output1 = best_model.predict_proba(dft[predictors].astype(float))[:,1]
output1 = (output1 > 0.5).astype(int)
output2 = best_model.predict(dft[predictors])
output2 = (output2 > 0.5).astype(int)
print np.mean(output1 - output2)

submission = pd.DataFrame({ 'PassengerId': dft['PassengerId'], 'Survived': output1 });

submission.to_csv('model_%f.csv' % np.max(np.mean(results, axis=1)), index=False)
