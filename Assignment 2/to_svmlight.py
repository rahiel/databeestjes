from __future__ import division

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import dump_svmlight_file

from sklearn.ensemble import RandomForestClassifier

def preprocess(df):
    d = df.isnull().sum().to_dict()
    items = sorted(d.items(), key=lambda kv: kv[1])

    # we only use attributes with less than 1000 missing values
    feature_filter = filter(lambda x: x[1] < 1000, items)
    feature_labels = [x[0] for x in feature_filter]

    feature_labels.remove("date_time")
    feature_labels.remove("srch_id")

    train = False
    if 'position' in feature_labels:
        train = True
    if train:
        feature_labels.remove("position")
        feature_labels.remove("click_bool")
        feature_labels.remove("booking_bool")

    dd = df.fillna(value=0)   # fill missing values
    features = dd[feature_labels].values
    qid = dd['srch_id'].values
    target = np.zeros(len(dd))
    if train:
        target = np.fmax((5*dd['booking_bool']).values, dd['click_bool'].values)

    return dd, features, qid, target, feature_labels

nrows = int(1E5)
data_train = pd.read_csv("training_set_VU_DM_2014.csv", header=0, nrows=2 * nrows)
data_test = pd.read_csv("test_set_VU_DM_2014.csv", header=0, nrows=nrows)
train, Xtr, qtr, ytr, feature_labels = preprocess(data_train[:nrows])
vali, Xva, qva, yva, feature_labels = preprocess(data_train[nrows:])
test, Xte, qte, yte, feature_labels = preprocess(data_test)
print "loaded files"

comment = ' '.join(map(lambda t: '%d:%s' % t, zip(range(len(feature_labels)), feature_labels)))

dump_svmlight_file(Xtr, ytr, 'spelen/train.svmlight', query_id = qtr, comment=comment)
dump_svmlight_file(Xva, yva, 'spelen/vali.svmlight', query_id = qva, comment=comment)
dump_svmlight_file(Xva, np.zeros(len(data_test)), 'spelen/test.svmlight', query_id = qte, comment=comment)
