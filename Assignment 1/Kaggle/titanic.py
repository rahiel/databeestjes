import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def convertSex(df):
    df['Sex'] = df['Sex'].map(lambda x: x == 'male').astype(int)

def fillAge(df):
    median_ages = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i,j] = df[(df.Sex == i) & (df.Pclass == j+1)]['Age'].dropna().median()

    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[ (df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j+1), 'Age'] = median_ages[i,j]

def fillFare(df):
    median_fares = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_fares[i,j] = df[(df.Sex == i) & (df.Pclass == j+1)]['Fare'].dropna().median()

    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[ (df.Fare.isnull()) & (df.Sex == i) & (df.Pclass == j+1), 'Fare'] = median_fares[i,j]

dft = pd.read_csv('test.csv', header=0)
df = pd.read_csv('train.csv', header=0)

convertSex(df)
convertSex(dft)
fillAge(df)
fillAge(dft)
fillFare(df)
fillFare(dft)

predictors = ['Pclass', 'SibSp', 'Parch', 'Sex', 'Age', 'Fare']

forest = RandomForestClassifier(n_estimators = 100)
forest.fit(df[predictors].values, df['Survived'])

output = forest.predict(dft[predictors].values)
output = (output > 0.5).astype(int)

submission = pd.DataFrame({
    'PassengerId': dft['PassengerId'], 'Survived': output
});

submission.to_csv('simplemodel.csv', index=False)
