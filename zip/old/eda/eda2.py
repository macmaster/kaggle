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
color=sns.color_palette()

train = pd.read_csv("train.csv", index_col="id")
test = pd.read_csv("test.csv", index_col="id")
data = pd.concat([train, test]) # all data
data.drop(["Y"], axis=1, inplace=True)
data = data.fillna(data.median())
    
# drop some collinear features
data.drop(["F26"], axis=1, inplace=True)

# junk features
data.drop(["F4", "F7", "F8", "F15", "F17", "F20", "F24"], axis=1, inplace=True)
data.drop(["F1", "F12", "F13"], axis=1, inplace=True) # further random forest selection
data.drop(["F9", "F16", "F21", "F5"], axis=1, inplace=True) # round 2 forest selection
data.drop(["F11", "F6"], axis=1, inplace=True) # round 3 extra forest selection
# data.drop(["F19", "F11", "F6", "F10"], axis=1, inplace=True) # round 3 extra forest selection

originalFeatures = data.columns
unskewedFeatures = [ "unskewed" + feat for feat in originalFeatures ]
robustFeatures = [ "robust" + feat for feat in originalFeatures ]
scaledFeatures = [ "scaled" + feat for feat in originalFeatures ]

# interactions
data = custom.get_interactions(data)

# transform feature names
def scaled_names(feats, skewedFeats, robustFeats, scaledFeats):
    newFeats = []
    for feat in feats:
        if (feat in skewedFeats):
            newFeats.append("unskewed" + feat)
        elif (feat in robustFeats):
            newFeats.append("robust" + feat)
        elif (feat in scaledFeats):
            newFeats.append("scaled" + feat)
        else:
            newFeats.append("scaled" + feat)
    return newFeats

# unskew some features with boxcox
skewedFeats = [
    "F2", "F14", "F25",             # hi numerical categorical features
    "F3", "F19", "F22", "F27",      # hi numerical quantitative features
]
robustFeats = ["F5", "F10"]
scaledFeats = list(set(data.columns) - set(skewedFeats) - set(robustFeats))

from scipy.stats import skew, boxcox
from sklearn.preprocessing import scale, robust_scale, minmax_scale
for feat in data:
    if (feat in skewedFeats):
        data["unskewed" + feat] = scale(boxcox(data[feat] + 1)[0])
    if (feat in scaledFeats):
        data["scaled" + feat] = scale(data[feat])
    if (feat in robustFeats):
        data["robust" + feat] = robust_scale(data[feat].values.reshape(-1, 1))
    data.drop(feat, axis=1, inplace=True)
data.fillna(data.median(), inplace=True)

print "data post processing: ", data.shape
xtest = data[train.shape[0]:]
xtrain = data[:train.shape[0]]
ytrain = train["Y"]

# # model features
# xgb_feats = scaled_names([ # interaction features to keep
#     "F18", "F3", "F25xF3", "F25xF27",  # F > 179
#     "F18xF19", "F14xF3", "F22xF27", "F14xF18", # "F19xF2", # F > 109 
#     "F18xF3", "F19xF3", "F18xF22", "F14xF19", # F > 97
#     "F2xF3", "F2xF27", "F22xF3", "F18xF2", "F14xF27", "F19", # F > 88
#     "F10xF22", "F19xF27", "F10xF18", "F18xF25", "F22xF25", # F > 75
#     "F3xF3", "F18xF27", "F19xF25", "F27", "F27xF3", "F22", # F > 65
#     "F10xF3", "F10xF27", "F18xF18", "F14xF22", "F19xF22", # F > 45
#     "F19xF19", "F10xF10", "F14", "F22xF22", "F10xF2", "F14xF2", # F > 21
#     # "F2", "F14xF14", "F10", "F2xF22", "F10xF25", "F14xF25", "F25", # F > 11
#     # "F2xF25", "F27xF27", "F2xF2", "F25xF25", "F10xF14"
# ], skewedFeats, robustFeats, scaledFeats)


# feats = scaled_names([ # interaction features to keep
#     "F18", "F3", "F25xF3", "F25xF27",  # F > 179
#     "F18xF19", "F14xF3", "F22xF27", "F14xF18", # "F19xF2", # F > 109 
#     "F18xF3", "F18xF22", # "F14xF19", # F > 97
#     "F2xF3", "F2xF27", "F14xF27", "F19", # F > 88
#     "F10xF18", "F22xF25", # F > 75
#     "F27", "F27xF3", "F14xF22",
# ], skewedFeats, robustFeats, scaledFeats)

# corrmat = xtrain[feats].corr()
corrmat = xtrain.corr()
ax = sns.heatmap(corrmat, vmin=-0.7, vmax=0.7, annot=True, fmt=".2f")
ax.set_title("Correlation Heatmap")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout()
# plt.savefig("%s/%s.png" % (PLOTDIR, "heatmap"))
plt.show()
plt.close()


# # classification scatter
# feats = set([(min(feat1, feat2), max(feat1, feat2)) 
#     for feat1 in xtrain for feat2 in xtrain[feats]])
# for feat1, feat2 in feats:
#     t, v = xtrain[feat1], xtrain[feat2]
#     f = t * v
#     plt.subplot(2, 1, 1)
#     plt.title("%s - %s" % (feat1, feat2))
#     plt.scatter(x=t, y=f, c=ytrain, edgecolor="r")
#     
#     plt.subplot(2, 1, 2)
#     plt.scatter(x=t, y=f, c=ytrain, edgecolor="r")
#     
#     plt.xlabel(feat1)
#     plt.show()
#     plt.close()

# sns.boxplot(y=xtrain["F18"], x=ytrain)
# # custom.count_plot(xtrain["F18sq"], ytrain)
# plt.title("f18 squared")
# plt.tight_layout()
# plt.show()
# plt.close()

# ## SMOTE to resample and balance distribution
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# resamp = RandomUnderSampler(random_state = 42)
# print "xtrain before smote: ", xtrain.shape
# print "ytrain before smote: ", ytrain.shape
# xcols = xtrain.columns
# xtrain, ytrain = resamp.fit_sample(xtrain, ytrain)
# xtrain = pd.DataFrame(xtrain, columns=xcols)
# print "xtrain after smote: ", xtrain.shape
# print "ytrain after smote: ", ytrain.shape
# sns.distplot(ytrain, kde=False)
# plt.title("Resampled Y")
# plt.show()
# plt.close()

# for feat in xtrain:
#     # custom.count_plot(data[feat].astype(int), ytrain)
#     sns.distplot(xtrain[feat], bins=50, kde=False)
#     plt.title(feat)
#     plt.show()
#     plt.close()

# # swarming and categorical plotting
# for feat in data[feats]:
#     print feat, ":", data[feat].dtype, data.shape
#     if data[feat].dtype == "int64":
#         # distribution
#         values = data[feat].astype(int)
#         custom.count_plot(values, y)
#         plt.ylabel("Y")
#         plt.title(feat) 
# 
#         plt.tight_layout()
#         plt.show()
#         plt.close()

# # skew experiments
# from sklearn.preprocessing import scale, RobustScaler
# from scipy.stats import skew, boxcox
# 
# data, y = xtrain, ytrain
# robustData = RobustScaler().fit_transform(data)
# robustData = pd.DataFrame(robustData, columns=data.columns)
# 
# for feat in data:
#     skewness = stats.skew(data[feat])
#     print feat, skewness
#     skewedData = data[feat]
#     scaledData = scale(data[feat])
#     
#     unskewedData = boxcox(data[feat] + 1)[0]
#     # unskewedData = scale(np.log1p(data[feat]))
#     data["unskewed" + feat] = unskewedData
#     data["robust" + feat] = robustData[feat]
# 
#     if (feat in packedFeats):
#         unskewedData = unskewedData.astype(int)
#         skewedData = skewedData.astype(int)
#         scaledData = scaledData.astype(int)
#         robustData[feat] = robustData[feat].astype(int)
# 
# 
#     k, nplots = 1, 4
#     plt.subplots(nplots, 1, figsize=(16,8))
#     
#     # # histogram
#     # plt.subplot(nplots, 1, k)
#     # plt.hist(skewedData, bins=50, facecolor="#440055", alpha=0.75)
#     # plt.title(feat + " histogram")
#     # plt.annotate("skewness: {0:.2f}".format(skew(skewedData)), 
#     #     xy=(0.8, 0.8), xycoords="axes fraction")
#     # k += 1
# 
#     # # unskewed histogram
#     # plt.subplot(nplots, 1, k)
#     # plt.hist(unskewedData, bins=50, facecolor="#440055", alpha=0.75)
#     # plt.title(feat + " unskewed histogram")
#     # plt.annotate("skewness: {0:.2f}".format(skew(unskewedData)), 
#     #     xy=(0.8, 0.8), xycoords="axes fraction")
#     # k += 1
# 
#     # # boxplot
#     # plt.subplot(nplots, 1, k)
#     # plt.boxplot(unskewedData)
#     # plt.title(feat + " boxplot")
#     # k += 1
# 
#     # skewed countplot
#     plt.subplot(nplots, 1, k)
#     custom.count_plot(skewedData, y)
#     plt.title("skewed countplot")
#     plt.annotate("skewness: {0:.2f}".format(skew(skewedData)), 
#         xy=(0.8, 0.8), xycoords="axes fraction")
#     k += 1
#     
#     # unit variance scaled
#     plt.subplot(nplots, 1, k)
#     custom.count_plot(scaledData, y)
#     plt.title("std scaled countplot")
#     plt.annotate("skewness: {0:.2f}".format(skew(scaledData)), 
#         xy=(0.8, 0.8), xycoords="axes fraction")
#     k += 1
# 
#     # robust scaled
#     plt.subplot(nplots, 1, k)
#     custom.count_plot(robustData[feat], y)
#     plt.title("robust scaled countplot")
#     plt.annotate("skewness: {0:.2f}".format(skew(robustData[feat])), 
#         xy=(0.8, 0.8), xycoords="axes fraction")
#     k += 1
# 
#     # unskewed countplot
#     plt.subplot(nplots, 1, k)
#     custom.count_plot(unskewedData, y)
#     plt.annotate("skewness: {0:.2f}".format(skew(unskewedData)), 
#         xy=(0.8, 0.8), xycoords="axes fraction")
#     plt.title("unskewed countplot")
#     k += 1
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
# feats = data.columns
# subset = pd.concat([data[unskewedFeatures + ["robustF5", "robustF7", "robustF8", "robustF10"]], y], axis=1)
# corrmat = subset.corr()
# ax = sns.heatmap(corrmat, vmin=-0.7, vmax=0.7, annot=True, fmt=".2f")
# ax.set_title("Correlation Heatmap")
# ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# plt.tight_layout()
# # plt.savefig("%s/%s.png" % (PLOTDIR, "heatmap"))
# plt.show()
# plt.close()

