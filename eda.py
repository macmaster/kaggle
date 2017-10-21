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
data = train.drop(["Y"], axis=1).fillna(train.median())
y = train["Y"]

# drop some collinear features
data.drop(["F23", "F26"], axis=1, inplace=True)

# # unskew some features with boxcox
# from scipy.stats import skew, boxcox
# from sklearn.preprocessing import scale
# data["F3"] = scale(boxcox(data["F3"] + 1)[0])
# data["F19"] = scale(boxcox(data["F19"] + 1)[0])
# data["F22"] = scale(boxcox(data["F22"] + 1)[0])
# data["F27"] = scale(boxcox(data["F27"] + 1)[0])

## # strong predictors and interactions
## data["F2xF14"] = scale(boxcox(data["F2"] * data["F14"] + 1)[0])
## data["F2xF25"] = scale(boxcox(data["F2"] * data["F25"] + 1)[0])
## data["F14xF25"] = scale(boxcox(data["F14"] * data["F25"] + 1)[0])
## 
## # numerical categorical features
## data["F2"] = scale(boxcox(data["F2"] + 1)[0])
## data["F14"] = scale(boxcox(data["F14"] + 1)[0])
## data["F25"] = scale(boxcox(data["F25"] + 1)[0])
## 
## # numerical quantitative features
## data["F3"] = scale(boxcox(data["F3"] + 1)[0])
## data["F19"] = scale(boxcox(data["F19"] + 1)[0])
## data["F22"] = scale(boxcox(data["F22"] + 1)[0])
## data["F27"] = scale(boxcox(data["F27"] + 1)[0])
## 
## data["F6"] = scale(boxcox(data["F6"] + 1)[0])
## data["F10"] = scale(boxcox(data["F10"] + 1)[0])

packedFeats = ["F3", "F6", "F9", "F17", "F19", "F27"]
junkFeats = ["F4", "F15", "F20", "F24"]
interactionFeats = ["F2", "F14", "F25", "F2xF14", "F2xF25", "F14xF25"]
mostFeats = data.columns.drop(packedFeats)
originalFeatures = data.columns
unskewedFeatures = [ "unskewed" + feat for feat in originalFeatures ]
robustFeatures = [ "robust" + feat for feat in originalFeatures ]

# PLOTDIR = "plots/univariate"
# # for feat in data[["F6", "F9", "F16", "F21"]]:
# for feat in data:
#     sns.distplot(data[feat].dropna(), bins=50, kde=False)
#     plt.title(feat)
#     # plt.xlim(0, 1000)
#     # plt.savefig("%s/%s.png" % (PLOTDIR, feat))
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

# skew experiments
from sklearn.preprocessing import scale, RobustScaler
from scipy.stats import skew, boxcox

robustData = RobustScaler().fit_transform(data)
robustData = pd.DataFrame(robustData, columns=data.columns)

for feat in data:
    skewness = stats.skew(data[feat])
    print feat, skewness
    skewedData = data[feat]
    scaledData = scale(data[feat])
    
    unskewedData = boxcox(data[feat] + 1)[0]
    # unskewedData = scale(np.log1p(data[feat]))
    data["unskewed" + feat] = unskewedData
    data["robust" + feat] = robustData[feat]

    if (feat in packedFeats):
        unskewedData = unskewedData.astype(int)
        skewedData = skewedData.astype(int)
        scaledData = scaledData.astype(int)
        robustData[feat] = robustData[feat].astype(int)


    k, nplots = 1, 4
    plt.subplots(nplots, 1, figsize=(16,8))
    
    # # histogram
    # plt.subplot(nplots, 1, k)
    # plt.hist(skewedData, bins=50, facecolor="#440055", alpha=0.75)
    # plt.title(feat + " histogram")
    # plt.annotate("skewness: {0:.2f}".format(skew(skewedData)), 
    #     xy=(0.8, 0.8), xycoords="axes fraction")
    # k += 1

    # # unskewed histogram
    # plt.subplot(nplots, 1, k)
    # plt.hist(unskewedData, bins=50, facecolor="#440055", alpha=0.75)
    # plt.title(feat + " unskewed histogram")
    # plt.annotate("skewness: {0:.2f}".format(skew(unskewedData)), 
    #     xy=(0.8, 0.8), xycoords="axes fraction")
    # k += 1

    # # boxplot
    # plt.subplot(nplots, 1, k)
    # plt.boxplot(unskewedData)
    # plt.title(feat + " boxplot")
    # k += 1

    # skewed countplot
    plt.subplot(nplots, 1, k)
    # custom.count_plot(skewedData, y)
    plt.title("skewed countplot")
    plt.annotate("skewness: {0:.2f}".format(skew(skewedData)), 
        xy=(0.8, 0.8), xycoords="axes fraction")
    k += 1
    
    # unit variance scaled
    plt.subplot(nplots, 1, k)
    # custom.count_plot(scaledData, y)
    plt.title("std scaled countplot")
    plt.annotate("skewness: {0:.2f}".format(skew(scaledData)), 
        xy=(0.8, 0.8), xycoords="axes fraction")
    k += 1

    # robust scaled
    plt.subplot(nplots, 1, k)
    # custom.count_plot(robustData[feat], y)
    plt.title("robust scaled countplot")
    plt.annotate("skewness: {0:.2f}".format(skew(robustData[feat])), 
        xy=(0.8, 0.8), xycoords="axes fraction")
    k += 1

    # unskewed countplot
    plt.subplot(nplots, 1, k)
    # custom.count_plot(unskewedData, y)
    plt.annotate("skewness: {0:.2f}".format(skew(unskewedData)), 
        xy=(0.8, 0.8), xycoords="axes fraction")
    plt.title("unskewed countplot")
    k += 1

    plt.tight_layout()
    # plt.show()
    plt.close()


# PLOTDIR = "plots/bivariate"
# feats = { (feat1, feat2) for feat1 in data for feat2 in data }
# for (feat1, feat2) in sorted(feats):
#     print "bivariate: ", (feat1, feat2)
#     sns.jointplot(x=feat1, y=feat2, data=data.dropna())
#     plt.savefig("%s/%s-%s.png" % (PLOTDIR, feat1, feat2))
#     plt.tight_layout()
#     plt.close()

PLOTDIR = "plots"
feats = data.columns
subset = pd.concat([data[unskewedFeatures + ["robustF5", "robustF7", "robustF8", "robustF10"]], y], axis=1)
corrmat = subset.corr()
ax = sns.heatmap(corrmat, vmin=-0.7, vmax=0.7, annot=True, fmt=".2f")
ax.set_title("Correlation Heatmap")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout()
# plt.savefig("%s/%s.png" % (PLOTDIR, "heatmap"))
plt.show()
plt.close()

