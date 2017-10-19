# model.py
# model for the competition
# author: Ronny Macmaster

import custom # custom function library
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
data.drop(["Y"], axis=1, inplace=True)
    
# # sanity check
# print data.head()
# print data.describe()

# preprocessing pipeline
print "data before pipeline: ", data.shape
columns = data.columns
data = data.fillna(data.median())
data = pd.DataFrame(data, columns=columns)

# unskew some features with boxcox
from scipy.stats import skew, boxcox
from sklearn.preprocessing import scale
data["F3"] = scale(boxcox(data["F3"] + 1)[0])
data["F19"] = scale(boxcox(data["F19"] + 1)[0])
data["F22"] = scale(boxcox(data["F22"] + 1)[0])
data["F27"] = scale(boxcox(data["F27"] + 1)[0])

# drop some collinear features
data.drop(["F23", "F26"], axis=1, inplace=True)

# # Feature Selection
# "F2", "F14", "F25", # skewed, but nicely correlated
# "F12", "F4", # interesting leverage points
data = data[["F3", "F14", "F18", "F19", "F22", "F27"]]
# data = data[["F3", "F14", "F18", "F19", "F27"]]
# data = data[["F3", "F11", "F18", "F19", "F22", "F27"]]

from sklearn.model_selection import train_test_split
xtest = data[train.shape[0]:]
xtrain, xtrain_val, ytrain, ytrain_val = train_test_split(
    data[:train.shape[0]], train["Y"], test_size=0.5, stratify=train["Y"]
)

# optional EDA
# custom.eda_countplot(xtrain, ytrain)

# # train a simple classifier
# from sklearn.linear_model import LogisticRegression
# lr_clf = LogisticRegression()
# lr_clf.fit(xtrain, ytrain)
# lr_clf_pred = lr_clf.predict(xtrain)

# train a random forest 
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier( n_jobs = 2,
    n_estimators = 60, criterion="entropy", max_features=None)
forest_clf.fit(xtrain, ytrain)
forest_pred = forest_clf.predict(xtrain)

print "Forest Feature Importances(%s):" % xtrain.columns.values
print forest_clf.feature_importances_

# # train an xgboost model
# import xgboost as xgb
# from sklearn.externals import joblib
# xgb_clf = xgb.XGBClassifier( nthread = 1,
#     n_estimators = 300, learning_rate=0.1
# )
# xgb_clf = xgb_clf.fit(
#     xtrain, ytrain,
#     eval_metric="auc",
# )
# xgb_pred = xgb_clf.predict(xtrain)
# joblib.dump(xgb_clf, "xgb_model.pkl")

# # plot xgb feature importance
# xgb_clf = joblib.load("xgb_model.pkl")
# xgb.plot_importance(xgb_clf)
# plt.title("XGB Feature Importance")
# plt.show()
# plt.close()

# evaluate performance metrics
from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix
clf = forest_clf # pick a classifier
clf_pred = clf.predict(xtrain) 
fpr, tpr, thresholds = roc_curve(ytrain, clf_pred)
roc_auc = auc(fpr, tpr)

print "confusion_matrix:\n", confusion_matrix(ytrain, clf_pred)
print "validation set score: ", roc_auc_score(ytrain_val, clf.predict(xtrain_val))
print "training set score: ", roc_auc_score(ytrain, clf.predict(xtrain))

# plot the roc curve
custom.roc_plot(fpr, tpr)
plt.show()
plt.close()

# submit solution
submit = pd.DataFrame()
submit["id"] = test.index
submit["Y"] = clf.predict_proba(xtest)[:, 1]
submit.to_csv("submit.csv", index=False)
