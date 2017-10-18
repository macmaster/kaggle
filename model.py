# model.py
# model for the competition
# author: Ronny Macmaster

import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import numpy as np

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

mpl.style.use(["fivethirtyeight"])
sns.set(style="whitegrid", color_codes=True)

train = pd.read_csv("train.csv", index_col="id")
test = pd.read_csv("test.csv", index_col="id")
data = pd.concat([train, test]) # all data

# # sanity check
# print data.head()
# print data.describe()

# manual preprocessing
print "data before preprocessing: ", data.shape
data.drop(["Y"], axis=1, inplace=True)

# unskew some features with boxcox
from scipy.stats import skew, boxcox
from sklearn.preprocessing import scale
data["F3"] = scale(boxcox(data["F3"].fillna(data["F3"].mean()) + 1)[0])
data["F19"] = scale(boxcox(data["F19"].fillna(data["F19"].mean()) + 1)[0])
data["F22"] = scale(boxcox(data["F22"].fillna(data["F22"].mean()) + 1)[0])
data["F27"] = scale(boxcox(data["F27"].fillna(data["F27"].mean()) + 1)[0])

# drop some collinear features
data.drop(["F23", "F26"], axis=1, inplace=True)


# preprocessing pipeline
print "data before pipeline: ", data.shape
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, StandardScaler, RobustScaler
columns = data.columns
pipeline = Pipeline([
    ("imputer", Imputer(strategy="mean")),
    ("scaler", RobustScaler()),
])

print data.columns
data = pipeline.fit_transform(data)
data = pd.DataFrame(data, columns=columns)

data = data[["F3", "F11", "F18", "F19", "F22", "F27"]]
xtest = data[train.shape[0]:]
xtrain = data[:train.shape[0]]
ytrain = train["Y"]

# # model eda
# for feat in data:
#     print "univariate: ", feat
#     sns.distplot(data[feat], kde=False)
#     plt.title(feat)
#     plt.show()
#     plt.close()

# train a simple classifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
clf = LogisticRegression()
clf.fit(xtrain, ytrain)
clf_pred = clf.decision_function(xtrain)

fpr, tpr, _ = roc_curve(ytrain, clf_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color="r", label="ROC Curve (area = %0.2f)" % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.legend(loc="lower right")
plt.show()
plt.close()


# # submit solution
# submit = pd.DataFrame()
# submit["id"] = test.index
# submit["Y"] = 0.0
# submit.to_csv("submit.csv", index=False)
