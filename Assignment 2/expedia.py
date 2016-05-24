from __future__ import division

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


nrows = int(1E6)
data = pd.read_csv("training_set_VU_DM_2014.csv", header=0, nrows=2 * nrows)
train = data[:nrows]
test = data[nrows:]
nrows = train.shape[0]


d = train.isnull().sum().to_dict()
items = sorted(d.items(), key=lambda kv: kv[1])

# we only use attributes with less than 1000 missing values
feature_filter = filter(lambda x: x[1] < 1000, items)
feature_labels = [x[0] for x in feature_filter]

feature_labels.remove("date_time")
train = train.fillna(value=0)   # fill missing values
test = test.fillna(value=0)   # fill missing values

features = train[feature_labels].values
target = train["booking_bool"].values
classifier = RandomForestClassifier(n_estimators=30)

classifier.fit(features, target)

predictions = classifier.predict_proba(test[feature_labels].values)[:, 1]
