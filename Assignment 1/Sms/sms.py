# step into the future
from __future__ import division

# standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# used in sanitization
from nltk.corpus import stopwords

# sklearn stuff
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.cross_validation import train_test_split as ttspl
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

# sklearn classifiers
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def convertToInt(df, col, dctGiven = None):
    if dctGiven is None:
        dct = {x:k for k,x in enumerate(df[col].unique())}
    else:
        dct = dctGiven

    df[col] = df[col].map(dct)

    return dct


# this call assumes an updated SmsCollection file, which is created from the
# original by executing (in vim):
#     :1s/label;text/label\ttext
#     :%s/^ham;/ham\t/g
#     :%s/^spam;/spam\t/g
sms = pd.read_csv('SmsCollectionTabDelimited.csv', '\t', header=0)
print convertToInt(sms, 'label')

# create frequency histograms from the text
vectorizer = TfidfVectorizer(min_df=2, stop_words='english')
freqs  = vectorizer.fit_transform(sms.text)
print freqs.shape

classifiers = [
    [DecisionTreeClassifier(), 
     {"max_features": [None, 100, 1000, 2500], "max_depth": [3, None]}],
    [LinearSVC(loss='hinge', random_state=42), 
        {"loss": ["hinge", "squared_hinge"]}],
    [SGDClassifier(loss='hinge', alpha=1e-3, random_state=42), 
     {"n_iter": [5, 25], "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}],
    [MultinomialNB(), {"alpha": [0, 0.5, 1, 2], "fit_prior": [True, False]}]
]
nrepeat = 11
results = np.zeros([len(classifiers), nrepeat])
for rs in range(nrepeat):
    idxTr, idxTe, lblTr, lblTe = ttspl(sms.index.values, sms.label, test_size=1/3, random_state=rs)

    maxacc = 0
    maxclf = 0
    for i, [clf, grid] in enumerate(classifiers):
        gs = GridSearchCV(clf, grid, cv=10)
        gs.fit(freqs[idxTr,:], lblTr)
        results[i, rs] = gs.score(freqs[idxTe,:], lblTe)
        print results

print np.mean(results, axis=1)
