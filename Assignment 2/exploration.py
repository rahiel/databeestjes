from __future__ import division

from collections import Counter
from dateutil.relativedelta import relativedelta
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors

train = pd.read_csv("training_set_VU_DM_2014.csv", header=0, parse_dates=[1])
nrows = train.shape[0]

# create a nice plot with x-axis the features, and y-axis the percentage of missing data
def barplot():
    fig, ax = plt.subplots(figsize=(1200/80, 500/80))
    d = train.isnull().sum().to_dict()
    items = sorted(d.items(), key=lambda kv: kv[1])

    bp = sns.barplot(map(lambda kv: kv[0], items),
                     map(lambda kv: 100*kv[1]/nrows, items),
                     palette = sns.color_palette('hls', n_colors=30))
    bp.set_ylabel("Percentage missing")

    for item in bp.get_xticklabels():
        item.set_rotation(90)

    plt.subplots_adjust(bottom=0.4)
    plt.savefig("barplot", dpi=400)

def outliers():
    cols = ['visitor_hist_adr_usd', 'price_usd', 'gross_bookings_usd', 'orig_destination_distance', 
            'comp1_rate_percent_diff',
            'comp2_rate_percent_diff',
            'comp3_rate_percent_diff',
            'comp4_rate_percent_diff',
            'comp5_rate_percent_diff',
            'comp6_rate_percent_diff',
            'comp7_rate_percent_diff',
            'comp8_rate_percent_diff']
    fig, axarr = plt.subplots(2, sharex=True, figsize=(1200/80, 500/80))
    #ax = train[cols].boxplot(rot=90, return_type="axes", sym='k.', showfliers=True)
    ax = sns.boxplot(train[cols], orient='h', whis=0.1, ax=axarr[0], fliersize=2)
    ax.set_xscale('log')
    ax = sns.boxplot(train[cols], orient='h', ax=axarr[1], fliersize=2)
    ax.set_xscale('log')
    plt.savefig("outliers", dpi=400)

def dateplot():
    months = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    def name(x):
        if x[2] == 15:
            return '%s %d' % (months[x[1]], x[0])
        else:
            return ''

    df2 = train[train.booking_bool == 1]

    dates = [(a.year, a.month, a.day) for a in df2.date_time]
    c = sorted(Counter(dates).items(), key=lambda kv: kv[0][0]*10000 + kv[0][1]*100 + kv[0][2])
    datesort = [kv[0][0]*10000 + kv[0][1]*100 + kv[0][2] for kv in c]
    names = [name(x) for x,y in c]
    fig, axarr = plt.subplots(2, sharex=True, figsize=(1200/80, 500/80))
    palette = sns.color_palette('hls', n_colors=12)
    clrs = [palette[d[0][1]-1] for d in c]
    bp1 = sns.barplot(datesort, [y for x,y in c], palette = clrs, ax=axarr[0], linewidth=0)
    axarr[0].set_xticklabels(names)

    startdates = [x+y for x,y in zip(df2.date_time, [relativedelta(days=x) for x in df2.srch_booking_window])]
    staydates = zip(startdates, df2.srch_length_of_stay)
    newdates = []
    for staydate in staydates:
        for day in range(staydate[1]):
            newdates.append(staydate[0] + relativedelta(days=day))
    newdates = [(a.year, a.month, a.day) for a in newdates if a.year < 2013 or (a.year == 2013 and a.month < 11)]
    newc = sorted(Counter(newdates).items(), key=lambda kv: kv[0][0]*10000 + kv[0][1]*100 + kv[0][2])
    newdatesort = [kv[0][0]*10000 + kv[0][1]*100 + kv[0][2] for kv in newc]
    newnames = [name(x) for x,y in newc]
    newclrs = [palette[d[0][1]-1] for d in newc]
    bp2 = sns.barplot(newdatesort, [y for x,y in newc], palette = newclrs, ax=axarr[1], linewidth=0)
    axarr[1].set_xticklabels(newnames)

    bp1.set_ylabel("Number of searches")
    bp2.set_ylabel("Number of active bookings")

    plt.savefig("dateplot", dpi=400)

def summarize():
    num_queries = len(train.srch_id.unique())
    num_hotel_countries = len(train.prop_country_id.unique())
    num_hotels = len(train.prop_id.unique())
    num_visitor_countries = len(train.visitor_location_country_id.unique())

    # the visitor countries and prop countries do not hash the same so same id does not imply same country!

    return num_queries, num_hotel_countries, num_hotels, num_visitor_countries


#barplot()
#outliers()
dateplot()
