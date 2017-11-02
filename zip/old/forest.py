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
columns = data.columns
data = data.fillna(data.median())
data = pd.DataFrame(data, columns=columns)

# drop some collinear features
data.drop(["F23", "F26"], axis=1, inplace=True)

# junk features
data.drop(["F4", "F7", "F8", "F15", "F17", "F20", "F24"], axis=1, inplace=True)
data.drop(["F1", "F12", "F13"], axis=1, inplace=True) # further random forest selection
data.drop(["F9", "F16", "F21", "F5"], axis=1, inplace=True) # round 2 forest selection

# interaction features
# data = custom.get_interactions(data, ["F3", "F27", "F18", "F19", "F14"])

# unskew some features with boxcox
from scipy.stats import skew, boxcox
from sklearn.preprocessing import scale, robust_scale, minmax_scale

skewedFeats = [
    "F2", "F14", "F25",             # hi numerical categorical features
    "F3", "F19", "F22", "F27",      # hi numerical quantitative features
    "F6", # "F10,                   # features the trees like
]
# skewedFeats = [] # turn off unskewing

robustFeats = [
    "F5", "F10",    
]
# robustFeats = [] # turn off robustscaling

scaledFeats = list(set(data.columns) - set(skewedFeats) - set(robustFeats))
# scaledFeats = [] # turn off scaling

for feat in data:
    if (feat in skewedFeats):
        data["unskewed" + feat] = scale(boxcox(data[feat] + 1)[0])
    if (feat in scaledFeats):
        data["scaled" + feat] = scale(data[feat])
    if (feat in robustFeats):
        data["robust" + feat] = robust_scale(data[feat].values.reshape(-1, 1))
    data.drop(feat, axis=1, inplace=True)

print "skewedFeats:\n", skewedFeats
print "robustFeats:\n", robustFeats
print "scaledFeats:\n", scaledFeats
data.fillna(data.median(), inplace=True)

# # strong predictors and interactions
# data["F2xF14"] = scale(boxcox(data["F2"] * data["F14"] + 1)[0])
# data["F2xF25"] = scale(boxcox(data["F2"] * data["F25"] + 1)[0])
# data["F14xF25"] = scale(boxcox(data["F14"] * data["F25"] + 1)[0])

# # Feature Selection
# data = data[["F2", "F3", "F14", "F18", "F19", "F22", "F27"]]
# data = data[["F3",  "F14", "F25", "F14xF25"]]
# data = data[["F3", "F14", "F18", "F19", "F27"]]
# data = data[["F3", "F11", "F18", "F19", "F22", "F27"]]

# features the forest likes:
# data = data[["F3", "F27", "F18", "F19", "F11", "F14", "F22", "F25", "F6"]]

from sklearn.model_selection import train_test_split
print "data post processing: ", data.shape
xtest = data[train.shape[0]:]
xtrain = data[:train.shape[0]]
ytrain = train["Y"]

# # build validation set
# xtrain, xtrain_val, ytrain, ytrain_val = train_test_split(
#     data[:train.shape[0]], train["Y"], test_size=0.5, stratify=train["Y"]
# )

# optional EDA
# custom.eda_countplot(xtrain, ytrain)

# train a random forest 
# best model params(1800):  {'max_features': 'log2', 'max_leaf_nodes': 300, 'criterion': 'entropy', 'min_samples_leaf': 92}
from sklearn.ensemble import RandomForestClassifier
rforest_clf = RandomForestClassifier( n_jobs = 2,
    n_estimators = 80, criterion="entropy", 
    min_samples_leaf=92, max_leaf_nodes=300,
    oob_score=True, max_features="log2")
rforest_clf.fit(xtrain, ytrain)
rforest_pred = rforest_clf.predict(xtrain)

# train an extra random forest
# best model params:  {'max_features': 'log2', 'max_leaf_nodes': 400, 'criterion': 'entropy', 'min_samples_leaf': 80}
from sklearn.ensemble import ExtraTreesClassifier
eforest_clf = ExtraTreesClassifier( n_jobs = 2,
    n_estimators = 800, criterion="entropy",
    min_samples_leaf=12, max_leaf_nodes=400,
    max_features="log2")
eforest_clf.fit(xtrain, ytrain)
eforest_pred = eforest_clf.predict(xtrain)

from sklearn.externals import joblib
joblib.dump(rforest_clf, "random_forest.pkl")
joblib.dump(eforest_clf, "extra_random_forest.pkl")

importances = pd.Series(rforest_clf.feature_importances_, index=xtrain.columns.values)
print "Random Forest Feature Importances:\n", importances.sort_values()
print "Random Forest oob score: ", rforest_clf.oob_score_

importances = pd.Series(eforest_clf.feature_importances_, index=xtrain.columns.values)
print "Extra Random Forest Feature Importances:\n", importances.sort_values()

rforest_clf = joblib.load("random_forest.pkl")
eforest_clf = joblib.load("extra_random_forest.pkl")

# grid search the xgb model (optmizes model hyperparameters)
from sklearn.model_selection import GridSearchCV
rforest_grid = {
    "max_features" : ["sqrt", "log2"],
    "min_samples_leaf" : [86, 92, 96],
    "max_leaf_nodes" : [300, 400],
    "criterion" : ["entropy"],
}

# eforest_grid = {
#     # "max_features" : ["sqrt", "log2"],
#     # "min_samples_leaf" : [12, 16, 18],
#     "n_estimators" : [200, 400, 800, 1200],
#     "max_features" : ["log2"],
#     "max_leaf_nodes" : [400],
#     "min_samples_leaf" : [12],
#     "criterion" : ["entropy"],
# }
# 
# grid_search = GridSearchCV(eforest_clf, eforest_grid, cv=5, verbose=150, scoring="roc_auc")
# grid_search.fit(xtrain, ytrain)
# print "best model params: ", grid_search.best_params_
# print "best model cv score: ", grid_search.best_score_

# evaluate performance metrics
from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix
clf = eforest_clf # pick a classifier
clf_pred = clf.predict_proba(xtrain)[:, 1]
fpr, tpr, thresholds = roc_curve(ytrain, clf_pred)
roc_auc = auc(fpr, tpr)

# model metrics and validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(clf, xtrain, ytrain, cv=8, verbose=150, scoring="roc_auc")
print "confusion_matrix:\n", confusion_matrix(ytrain, clf.predict(xtrain))
print "training set score: ", roc_auc_score(ytrain, clf_pred)
print "cross validation scores:\n", cv_scores
print "cv stats (mean, std): (%f, %f)" % (cv_scores.mean(), cv_scores.std())
# print "validation set score: ", roc_auc_score(ytrain_val, clf.predict(xtrain_val))

# plot the roc curve
custom.roc_plot(fpr, tpr)
plt.show()
plt.close()

# submit solution
submit = pd.DataFrame()
submit["id"] = test.index
submit["Y"] = clf.predict_proba(xtest)[:, 1]
submit.to_csv("eforest.csv", index=False)
