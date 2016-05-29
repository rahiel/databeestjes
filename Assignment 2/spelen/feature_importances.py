from rankpy.models import LambdaMART
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open("test.svmlight") as f:
    head = [next(f) for x in range(4)]
features = map(lambda x: x.split(':')[1], head[-1][2:-1].split(' '))
columns = pd.read_csv("../test_set_VU_DM_2014.csv", header=0, nrows = 1).columns.values.tolist()
lm = LambdaMART.load("LambdaMartModel0.5.model")
feats = dict(zip(features, lm.feature_importances()))
feats = sorted(feats.items(), key=lambda kv: -kv[1])

fig, ax = plt.subplots(figsize=(1200/120, 500/120))
bp = sns.barplot( map(lambda x: x[0], feats), map(lambda x: x[1], feats))

for item in bp.get_xticklabels():
    item.set_rotation(90)
plt.subplots_adjust(bottom=0.5)

plt.savefig("feature_importances", dpi=400)
