# eda.py
# explore the data
# author: Ronny Macmaster

import custom
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import numpy as np
import scipy as sp

mpl.style.use(["fivethirtyeight"])
sns.set(style="whitegrid", color_codes=True)

train = pd.read_csv("train.csv", index_col="id")
data = train.drop(["Y"], axis=1)
y = train["Y"]

# drop some collinear features
data.drop(["F23", "F26"], axis=1, inplace=True)

# unskew some features with boxcox
from scipy.stats import skew, boxcox
from sklearn.preprocessing import scale
# data["F3"] = boxcox(data["F3"].fillna(0) + 1)[0]
# data["F19"] = boxcox(data["F19"].fillna(0) + 1)[0]
# data["F22"] = scale(boxcox(data["F22"].fillna(data["F22"].median()) + 1)[0])
data["F27"] = boxcox(data["F27"].fillna(0) + 1)[0]

# # preprocessing pipeline
# print "data before pipeline: ", data.shape
# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.preprocessing import Imputer, StandardScaler, RobustScaler
# columns = data.columns
# pipeline = Pipeline([
#     ("imputer", Imputer(strategy="median")),
#     # ("scaler", RobustScaler()),
# ])

# print data.columns
# data = pipeline.fit_transform(data)
# data = pd.DataFrame(data, columns=columns)

# PLOTDIR = "plots/univariate"
# # for feat in data[["F6", "F9", "F16", "F21"]]:
# for feat in data:
#     sns.distplot(data[feat].dropna(), bins=50, kde=False)
#     plt.title(feat)
#     # plt.xlim(0, 1000)
#     # plt.savefig("%s/%s.png" % (PLOTDIR, feat))
#     plt.show()
#     plt.close()


# swarming and categorical plotting
for feat in data[["F14"]]:
    print feat, ":", data[feat].dtype, data.shape
    if data[feat].dtype == "int64":
        # distribution
        values = data[feat].dropna().astype(int)
        custom.count_plot(values, y)
        plt.ylabel("Y")
        plt.title(feat) 

        plt.tight_layout()
        plt.show()
        plt.close()

# # skew experiments
# from sklearn.preprocessing import scale
# from scipy.stats import skew, boxcox
# PLOTDIR = "plots/unibox"
# # feats = ["F5", "F6", "F9", "F10", "F16", "F20", "F21"]
# # feats = ["F3", "F19", "F22", "F27"]
# feats = data.columns
# for feat in data[feats]:
#     skewness = stats.skew(data[feat])
#     skewedData = scale(data[feat])
#     unskewedData = scale(boxcox(data[feat] + 1)[0])
#     # unskewedData = scale(np.log1p(data[feat]))
#     data["unskewed" + feat] = unskewedData
#     
#     # histogram
#     plt.subplot(3, 1, 1)
#     plt.hist(skewedData, bins=500, facecolor="#440055", alpha=0.75)
#     plt.title(feat + " histogram")
#     plt.annotate("skewness: {0:.2f}".format(skew(skewedData)), 
#         xy=(0.8, 0.8), xycoords="axes fraction")
# 
#     # unskewed histogram
#     plt.subplot(3, 1, 2)
#     plt.hist(unskewedData, bins=500, facecolor="#440055", alpha=0.75)
#     plt.title(feat + " unskewed histogram")
#     plt.annotate("skewness: {0:.2f}".format(skew(unskewedData)), 
#         xy=(0.8, 0.8), xycoords="axes fraction")
# 
#     # boxplot
#     plt.subplot(3, 1, 3)
#     plt.boxplot(unskewedData)
#     plt.title(feat + " boxplot")
# 
#     plt.tight_layout()
#     plt.show()
#     plt.close()


# PLOTDIR = "plots/bivariate"
# feats = { (feat1, feat2) for feat1 in data for feat2 in data }
# for (feat1, feat2) in sorted(feats):
#     print "bivariate: ", (feat1, feat2)
#     sns.jointplot(x=feat1, y=feat2, data=data.dropna())
#     plt.savefig("%s/%s-%s.png" % (PLOTDIR, feat1, feat2))
#     plt.tight_layout()
#     plt.close()

# PLOTDIR = "plots"
# feats = ["F2", "F3", "F5", "F10", "F14", "F18", "F25", "Y", "unskewedF3", "unskewedF19", "unskewedF22", "unskewedF27"]
# subset = data[feats]
# ax = sns.heatmap(subset.corr(), vmin=-0.7, vmax=0.7, annot=True)
# ax.set_title("Correlation Heatmap")
# ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# plt.savefig("%s/%s.png" % (PLOTDIR, "heatmap"))
# plt.show()
# plt.close()
# 
