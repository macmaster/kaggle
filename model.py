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

# interaction features
# data = custom.get_interactions(data, ["F3", "F27", "F18", "F19", "F14"])

# unskew some features with boxcox
from scipy.stats import skew, boxcox
from sklearn.preprocessing import scale

# # strong predictors and interactions
data["F2xF14"] = scale(boxcox(data["F2"] * data["F14"] + 1)[0])
data["F2xF25"] = scale(boxcox(data["F2"] * data["F25"] + 1)[0])
data["F14xF25"] = scale(boxcox(data["F14"] * data["F25"] + 1)[0])

# numerical categorical features
data["F2"] = scale(boxcox(data["F2"] + 1)[0])
data["F14"] = scale(boxcox(data["F14"] + 1)[0])
data["F25"] = scale(boxcox(data["F25"] + 1)[0])

# numerical quantitative features
data["F3"] = scale(boxcox(data["F3"] + 1)[0])
data["F19"] = scale(boxcox(data["F19"] + 1)[0])
data["F22"] = scale(boxcox(data["F22"] + 1)[0])
data["F27"] = scale(boxcox(data["F27"] + 1)[0])

# data["F3xF25"] = boxcox(data["F3"] * data["F25"] + 1)[0]
# data["F3xF27"] = boxcox(data["F3"] * data["F27"] + 1)[0]
# data["F25xF27"] = boxcox(data["F25"] * data["F27"] + 1)[0]

# # Feature Selection
# "F2", "F14", "F25", # skewed, but nicely correlated
# "F12", "F4", # interesting leverage points
# data = data[["F2", "F3", "F14", "F18", "F19", "F22", "F27"]]
# data = data[["F3",  "F14", "F25", "F14xF25"]]
# data = data[["F3", "F14", "F18", "F19", "F27"]]
# data = data[["F3", "F11", "F18", "F19", "F22", "F27"]]


# only scale features
data["F11"] = scale(data["F11"])
data["F18"] = scale(data["F18"])

# features the forest likes:
data["F6"] = scale(boxcox(data["F6"] + 1)[0])
# data = data[["F3", "F27", "F18", "F19", "F11", "F14", "F22", "F25", "F6"]]

# features the xgb likes:
data["F10"] = scale(boxcox(data["F10"] + 1)[0])
data = data[["F3", "F19", "F2", "F27", "F18", "F22", "F25", "F10", "F6", "F11", "F14"]]


from sklearn.model_selection import train_test_split
print "data post processing: ", data.shape
xtest = data[train.shape[0]:]
xtrain = data[:train.shape[0]]
ytrain = train["Y"]
# xtrain, xtrain_val, ytrain, ytrain_val = train_test_split(
#     data[:train.shape[0]], train["Y"], test_size=0.5, stratify=train["Y"]
# )

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
    n_estimators = 20, criterion="entropy", 
    oob_score=True, max_features="sqrt")
forest_clf.fit(xtrain, ytrain)
forest_pred = forest_clf.predict(xtrain)

importances = pd.Series(forest_clf.feature_importances_, index=xtrain.columns.values)
print "Forest Feature Importances:\n", importances.sort_values()
print "Forest oob score: ", forest_clf.oob_score_

# train an xgboost model
import xgboost as xgb
from sklearn.externals import joblib
xgb_clf = xgb.XGBClassifier( nthread = 1,
    n_estimators = 600, max_depth=6,
    learning_rate=0.001
)
xgb_clf = xgb_clf.fit(
    xtrain, ytrain,
    eval_metric="auc",
)
xgb_pred = xgb_clf.predict(xtrain)
joblib.dump(xgb_clf, "xgb_model.pkl")

# plot xgb feature importance
xgb_clf = joblib.load("xgb_model.pkl")
xgb.plot_importance(xgb_clf)
plt.title("XGB Feature Importance")
plt.show()
plt.close()

# grid search the xgb model (optmizes model hyperparameters)
from sklearn.model_selection import GridSearchCV
xgb_grid = {
    "n_estimators" : [300, 600, 1000],
    "max_depth" : [2, 4, 8],
    "learning_rate" : [0.01, 0.001],
}

grid_search = GridSearchCV(xgb_clf, xgb_grid, cv=5, scoring="roc_auc")
grid_search.fit(xtrain, ytrain)
# print "best xgb grid params: ", grid_search.best_params_



# evaluate performance metrics
from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix
clf = xgb_clf # pick a classifier
clf_pred = clf.predict(xtrain) 
fpr, tpr, thresholds = roc_curve(ytrain, clf_pred)
roc_auc = auc(fpr, tpr)

# model metrics and validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(clf, xtrain, ytrain, cv=5, scoring="roc_auc")
print "confusion_matrix:\n", confusion_matrix(ytrain, clf_pred)
print "validation set score: ", roc_auc_score(ytrain_val, clf.predict(xtrain_val))
print "training set score: ", roc_auc_score(ytrain, clf.predict(xtrain))
print "cross validation scores:\n", cv_scores
print "cv stats (mean, std): (%f, %f)" % (cv_scores.mean(), cv_scores.std())

# plot the roc curve
custom.roc_plot(fpr, tpr)
plt.show()
plt.close()

# submit solution
submit = pd.DataFrame()
submit["id"] = test.index
submit["Y"] = clf.predict_proba(xtest)[:, 1]
submit.to_csv("submit.csv", index=False)
