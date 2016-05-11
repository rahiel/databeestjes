from __future__ import division

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

nrows = 1E5
train = pd.read_csv("training_set_VU_DM_2014.csv", header=0, nrows=nrows)
nrows = train.shape[0]

# create a nice plot with x-axis the features, and y-axis the percentage of missing data
def barplot():
    d = train.isnull().sum().to_dict()
    items = sorted(d.items(), key=lambda kv: kv[1])

    bp = sns.barplot(map(lambda kv: kv[0], items), 
                     map(lambda kv: 100*kv[1]/nrows, items), 
                     palette = sns.husl_palette(30, s=1))
    bp.set_ylabel("Percentage missing")

    for item in bp.get_xticklabels():
        item.set_rotation(90)

    plt.subplots_adjust(bottom=0.4)
    plt.show()

def summarize():
    num_queries = len(train.srch_id.unique())
    num_hotel_countries = len(train.prop_country_id.unique())
    num_hotels = len(train.prop_id.unique())
    num_visitor_countries = len(train.visitor_location_country_id.unique())

    # the visitor countries and prop countries do not hash the same so same id does not imply same country!

    return num_queries, num_hotel_countries, num_hotels, num_visitor_countries


barplot()
