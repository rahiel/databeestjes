from __future__ import division

from multiprocessing import Pool

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import dump_svmlight_file

from sklearn.ensemble import RandomForestClassifier


def preprocess(df, train):
    feature_labels = df.columns.values.tolist()

    # we remove the features where more than 50% of the data is missing from the training set
    remove = []
    for n in range(1, 9):
        remove += ["comp%d_inv" % n, "comp%d_rate" % n, "comp%d_rate_percent_diff" % n]
    remove += ["srch_query_affinity_score", "visitor_hist_adr_usd", "visitor_hist_starrating"]

    if "position" in feature_labels:
        remove += ["position", "click_bool", "booking_bool", "gross_bookings_usd"]  # training set only

    remove += ["srch_id", "prop_id"]

    for l in remove:
        feature_labels.remove(l)

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
    df_datetime = pd.DatetimeIndex(df.date_time)
    df["month"] = df_datetime.month
    df["week"] = df_datetime.week
    df["day"] = df_datetime.dayofweek + 1
    df["hour"] = df_datetime.hour
    feature_labels += ["month", "week", "day", "hour"]
    feature_labels.remove("date_time")

    # sanitize prop_log_historical_price
    df["prop_historical_price"] = (np.e ** df["prop_log_historical_price"]).replace(1.0, 10000)
    feature_labels.append("prop_historical_price")
    feature_labels.remove("prop_log_historical_price")

    features = df[feature_labels].values
    qid = df['srch_id'].values
    target = np.zeros(len(df))
    if 'booking_bool' in df.columns.values.tolist():
        target = np.fmax((5 * df['booking_bool']).values, df['click_bool'].values)

    return df, features, qid, target, feature_labels


nrows = None
#nrows = int(1e5)
data_train = pd.read_csv("training_set_VU_DM_2014.csv", header=0, parse_dates=[1], nrows=nrows)
try:
    data_test = pd.read_csv("testsetnew.csv", header=0, parse_dates=[1], nrows=nrows)
except IOError:
    data_test = pd.read_csv("test_set_VU_DM_2014.csv", header=0, parse_dates=[1], nrows=nrows)

print("loaded csv's")

# pre-fill missing prop_location_score2 scores with first quartile of country:
# source Bing Xu et al (fourth place)
all_data = pd.concat([data_train, data_test], copy=False)
location_quartile = all_data.groupby("prop_country_id")["prop_location_score2"].quantile(q=0.25)

for d in (data_train, data_test):
    d["prop_location_score2_quartile"] = location_quartile[d.prop_id].values
    d["prop_location_score2"].fillna(d["prop_location_score2_quartile"])
    del d["prop_location_score2_quartile"]


# fill missing values with worst case scenario. Source: Jun Wang 3rd place
# ["prop_review_score", "prop_location_score2", "orig_destination_distance"]
data_train = data_train.fillna(value=-1)
data_test = data_test.fillna(value=-1)

# feature engineering using all numeric features
# avg/median/std numeric features per prop_id
numeric_features = ["prop_starrating", "prop_review_score", "prop_location_score1", "prop_location_score2"]
all_data = pd.concat([data_train, data_test], copy=False)

for label in numeric_features:
    mean = all_data.groupby("prop_id")[label].mean().fillna(value=-1)
    median = all_data.groupby("prop_id")[label].median().fillna(value=-1)
    std = all_data.groupby("prop_id")[label].std().fillna(value=-1)

    for d in (data_train, data_test):
        d[label + "_mean"] = mean[d.prop_id].values
        d[label + "_median"] = median[d.prop_id].values
        d[label + "_std"] = std[d.prop_id].values


train, Xtr, qtr, ytr, feature_labels = preprocess(data_train[data_train.srch_id % 10 != 0], train=True)
print("preprocessed training data")
vali, Xva, qva, yva, feature_labels = preprocess(data_train[data_train.srch_id % 10 == 0], train=True)
del data_train
print("preprocessed validation data")
test, Xte, qte, yte, feature_labels = preprocess(data_test, train=False)
print("preprocessed test data")
del data_test

comment = ' '.join(map(lambda t: '%d:%s' % t, zip(range(len(feature_labels)), feature_labels)))


def dump(args):
    """Dumps to svmlight format."""
    x, y, filename, query_id, comment = args
    dump_svmlight_file(x, y, filename, query_id=query_id, comment=comment, zero_based=False)

p = Pool()
p.map(dump, ((Xtr, ytr, 'spelen/train_without_means.svmlight', qtr, comment),
             (Xva, yva, 'spelen/vali_without_means.svmlight', qva, comment),
             (Xte, yte, 'spelen/test_without_means.svmlight', qte, comment)))
