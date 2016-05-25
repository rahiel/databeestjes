from __future__ import division

from collections import Counter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors

nrows = 1E5
train = pd.read_csv("training_set_VU_DM_2014.csv", header=0, nrows=nrows, parse_dates=[1])
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

def dateplot():
    months = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    def name(x):
        if x[2] == 1:
            return '%s 1, %d' % (months[x[1]], x[0])
        else:
            return ''

    dates = [(a.year, a.month, a.day) for a in train.date_time]
    c = sorted(Counter(dates).items(), key=lambda kv: kv[0][0]*10000 + kv[0][1]*100 + kv[0][2])
    datesort = [kv[0][0]*10000 + kv[0][1]*100 + kv[0][2] for kv in c]
    names = [name(x) for x,y in c]
    fig, ax = plt.subplots()
    #plt.hist([y for x,y in c], bins=len(c), )
    palette = ["#ffffff","#cfffff","#9fffff","#70f3ff","#43c3ff","#1694ff","#0065e7","#0038b7","#000b88","#00005a","#00002d","#000000"]
    palette =["black","green","red","blue","magenta","gray","lightblue","lime","pink","yellow","peachpuff","lightsalmon"]
    palette = sns.color_palette(n_colors=12)
    print([d[0][1]-1 for d in c])
    clrs = [palette[d[0][1]-1] for d in c]
    # colors = [ for p in palette]
    # print(clrs[:3])
    index = np.arange(len(datesort))
    sns.barplot(datesort, [y for x,y in c], palette = clrs)
    # sns.bar(datesort, [y for x,y in c], color=clrs)
    ax.set_ticksize(5)
    ax.set_xticklabels(names)
    plt.show()

def summarize():
    num_queries = len(train.srch_id.unique())
    num_hotel_countries = len(train.prop_country_id.unique())
    num_hotels = len(train.prop_id.unique())
    num_visitor_countries = len(train.visitor_location_country_id.unique())

    # the visitor countries and prop countries do not hash the same so same id does not imply same country!

    return num_queries, num_hotel_countries, num_hotels, num_visitor_countries


dateplot()
