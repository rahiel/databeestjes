from __future__ import division

from multiprocessing import Pool

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import dump_svmlight_file

from sklearn.ensemble import RandomForestClassifier


def preprocess(df):
    feature_labels = df.columns.values.tolist()

    # we remove the features where more than 50% of the data is missing from the training set
    remove = []
    for n in range(1, 9):
        remove += ["comp%d_inv" % n, "comp%d_rate" % n, "comp%d_rate_percent_diff" % n]
    remove += ["srch_query_affinity_score", "visitor_hist_adr_usd", "visitor_hist_starrating"]

    train = False
    if "position" in feature_labels:
        train = True
        remove += ["position", "click_bool", "booking_bool", "gross_bookings_usd"]  # training set only

    for l in remove:
        feature_labels.remove(l)

    # fill missing values with worst case scenario. Source: Jun Wang 3rd place
    # ["prop_review_score", "prop_location_score2", "orig_destination_distance"]
    df = df.fillna(value=-1)

    # outliers in hotel price, hotels with price > 10000 are removed from training set. Source: David Wind
    if train:
        df = df[df["price_usd"] < 10000]

    #######################
    # FEATURE ENGINEERING #
    #######################

    # From paper "Combination of Diverse Ranking Models for Personalized Expedia Hotel Searches"
    df["count_window"] = df["srch_room_count"] * max(df["srch_booking_window"]) + df["srch_booking_window"]
    feature_labels.append("count_window")

    # month, week, day of the week and hour of search
    df["month"] = [df["date_time"].iloc[i].month for i in range(len(df))]       # jan: 1, dec:12
    df["week"] = [df["date_time"].iloc[i].week for i in range(len(df))]         # weeks from 1 - 52/53
    df["day"] = [df["date_time"].iloc[i].isoweekday() for i in range(len(df))]  # mon: 1, sun:7
    df["hour"] = [df["date_time"].iloc[i].hour for i in range(len(df))]         # range(24)
    feature_labels += ["month", "week", "day", "hour"]
    feature_labels.remove("date_time")

    # avg, mean, std per hotel. from 1st place leaderboard TODO

    features = df[feature_labels].values
    qid = df['srch_id'].values
    target = np.zeros(len(df))
    if train:
        target = np.fmax((5 * df['booking_bool']).values, df['click_bool'].values)

    return df, features, qid, target, feature_labels


data_train = pd.read_csv("training_set_VU_DM_2014.csv", header=0, parse_dates=[1])
data_test = pd.read_csv("test_set_VU_DM_2014.csv", header=0, parse_dates=[1])
print("loaded csv's")
train, Xtr, qtr, ytr, feature_labels = preprocess(data_train[data_train.srch_id % 10 != 0])
print("preprocessed training data")
vali, Xva, qva, yva, feature_labels = preprocess(data_train[data_train.srch_id % 10 == 0])
print("preprocessed validation data")
test, Xte, qte, yte, feature_labels = preprocess(data_test)
print("preprocessed test data")


comment = ' '.join(map(lambda t: '%d:%s' % t, zip(range(len(feature_labels)), feature_labels)))


def dump(args):
    """Dumps to svmlight format."""
    x, y, filename, query_id, comment = args
    dump_svmlight_file(x, y, filename, query_id=query_id, comment=comment)

p = Pool()
# dump_svmlight_file(Xtr, ytr, 'spelen/train.svmlight', query_id=qtr, comment=comment)
# dump_svmlight_file(Xva, yva, 'spelen/vali.svmlight', query_id=qva, comment=comment)
# dump_svmlight_file(Xte, np.zeros(len(data_test)), 'spelen/test.svmlight', query_id=qte, comment=comment)
p.map(dump, ((Xtr, ytr, 'spelen/train.svmlight', qtr, comment),
             (Xva, yva, 'spelen/vali.svmlight', qva, comment),
             (Xte, np.zeros(len(data_test)), 'spelen/test.svmlight', qte, comment)))
