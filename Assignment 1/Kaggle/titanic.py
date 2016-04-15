from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn import cross_validation, metrics
from sklearn.cross_validation import train_test_split

# fill the NaN-values in the given column with the non-NaN-median
def fillMedian(df, col, medianGiven = None):
    if medianGiven is None:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(medianGiven)

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

""" Data enchrichment phase """

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

df['FamilySize'] = df['SibSp'] + df['Parch']
dft['FamilySize'] = dft['SibSp'] + dft['Parch']

# choose which columns to predict with
predictors = ['Pclass', 'SibSp', 'Parch', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']

""" Training phase """

traindata = df[predictors].values
targets = df.Survived

Xtr, Xte, ytr, yte = train_test_split( traindata, targets, test_size = 1/3, random_state=42)
print 

# train classifier
forest = RandomForestClassifier(n_estimators = 100)
"""
scores = cross_validation.cross_val_score(forest, df[predictors].values, df.Survived, cv=5)
print scores.mean()
"""

forest.fit(Xtr, ytr)
predictions = forest.predict(Xte)
print metrics.classification_report(yte, predictions)

""" Testing phase """

# predict final output
output = forest.predict(dft[predictors].values)
output = (output > 0.5).astype(int)

submission = pd.DataFrame({ 'PassengerId': dft['PassengerId'], 'Survived': output });

submission.to_csv('simplemodel_embarked.csv', index=False)
