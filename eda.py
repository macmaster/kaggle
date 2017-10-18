# eda.py
# explore the data
# author: Ronny Macmaster

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy as sp

mpl.style.use(["fivethirtyeight"])
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv("train.csv", index_col="id")
print data.head()
print data.describe()

# PLOTDIR = "plots/univariate"
# for feat in data:
#     print "univariate: ", feat
#     sns.distplot(data[feat].dropna(), kde=False)
#     plt.title(feat)
#     plt.savefig("%s/%s.png" % (PLOTDIR, feat))
#     plt.close()

# PLOTDIR = "plots/bivariate"
# feats = { (feat1, feat2) for feat1 in data for feat2 in data }
# for (feat1, feat2) in sorted(feats):
#     print "bivariate: ", (feat1, feat2)
#     sns.jointplot(x=feat1, y=feat2, data=data.dropna())
#     plt.savefig("%s/%s-%s.png" % (PLOTDIR, feat1, feat2))
#     plt.tight_layout()
#     plt.close()

PLOTDIR = "plots"
feats = ["F2", "F3", "F5", "F14", "F18", "F23", "F25", "F26", "Y"]
subset = data[feats].corr()
ax = sns.heatmap(subset, vmin=-0.7, vmax=0.7, annot=True)
ax.set_title("Correlation Heatmap")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.savefig("%s/%s.png" % (PLOTDIR, "heatmap"))
plt.show()
plt.close()

