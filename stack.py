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

print "data post processing: ", data.shape
xtest = data[train.shape[0]:]
xtrain = data[:train.shape[0]]
ytrain = train["Y"]

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
rforest_pred = rforest_clf.predict_proba(xtrain)[:, 1]
rforest_test_pred = rforest_clf.predict_proba(xtest)[:, 1]

# train an extra random forest
# best model params(800):  {'max_features': 'log2', 'max_leaf_nodes': 400, 'criterion': 'entropy', 'min_samples_leaf': 12}
from sklearn.ensemble import ExtraTreesClassifier
eforest_clf = ExtraTreesClassifier( n_jobs = 2,
    n_estimators = 80, criterion="entropy",
    min_samples_leaf=12, max_leaf_nodes=400,
    max_features="log2")
eforest_clf.fit(xtrain, ytrain)
eforest_pred = eforest_clf.predict_proba(xtrain)[:, 1]
eforest_test_pred = eforest_clf.predict_proba(xtest)[:, 1]

from sklearn.externals import joblib
joblib.dump(rforest_clf, "models/random_forest.pkl")
joblib.dump(eforest_clf, "models/extra_random_forest.pkl")

importances = pd.Series(rforest_clf.feature_importances_, index=xtrain.columns.values)
print "Random Forest Feature Importances:\n", importances.sort_values()
print "Random Forest oob score: ", rforest_clf.oob_score_

importances = pd.Series(eforest_clf.feature_importances_, index=xtrain.columns.values)
print "Extra Random Forest Feature Importances:\n", importances.sort_values()

rforest_clf = joblib.load("models/random_forest.pkl")
eforest_clf = joblib.load("models/extra_random_forest.pkl")

# # grid search the forest model (optmizes model hyperparameters)
# from sklearn.model_selection import GridSearchCV
# rforest_grid = {
#     "max_features" : ["sqrt", "log2"],
#     "min_samples_leaf" : [86, 92, 96],
#     "max_leaf_nodes" : [300, 400],
#     "criterion" : ["entropy"],
# }
# 
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


###### Forest Stack
# run a stacking generalization over the 2 forest models
# build validation set
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.base import clone
xtrain_stack, xval_stack, ytrain_stack, yval_stack = train_test_split(
    data[:train.shape[0]], train["Y"], test_size=0.3, stratify=train["Y"]
)

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
class StackedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clfs, gen, folds=5, shuffle=False):
        self.skf = StratifiedKFold(n_splits=folds, shuffle=shuffle)
        self.clfs = clfs # stacked classifiers
        self.gen = gen # level1 generalizer
        self.folds = folds

    def fit(self, X, y):
        ytest = pd.Series()
        level1 = pd.DataFrame()
        for train_index, test_index in self.skf.split(X, y):
            xtrain, xtest = X[train_index], X[test_index]
            ytrain = y[train_index]
            ytest = pd.concat([ytest, pd.Series(y[test_index])], axis=0, ignore_index=True)
            
            # train level 0
            level0 = pd.DataFrame()
            for clf in self.clfs:
                clf.fit(xtrain, ytrain)
                clf_pred = pd.DataFrame(clf.predict_proba(xtest)[:, 1])
                level0 = pd.concat([level0, clf_pred], axis=1, ignore_index=True)

            level1 = pd.concat([level1, level0], axis=0, ignore_index=True)

        print "level1:", level1.shape, "\n", level1.head()
        print "ytest:", y.shape, "\n", y.head()
        self.gen.fit(level1, ytest)
        return self

    def predict_proba(self, X):
        # predict level 0
        level1 = pd.DataFrame()
        for clf in self.clfs:
            clf_pred = pd.DataFrame(clf.predict_proba(X)[:, 1])
            level1 = pd.concat([level1, clf_pred], axis=1, ignore_index=True)

        return self.gen.predict_proba(level1)



# warning, will unfit previous models
print "Training makeshift stack."
rforest_stack_clf = clone(rforest_clf).fit(xtrain_stack, ytrain_stack)
eforest_stack_clf = clone(eforest_clf).fit(xtrain_stack, ytrain_stack)
rforest_stack_pred = pd.DataFrame(rforest_stack_clf.predict_proba(xval_stack)[:, 1])
eforest_stack_pred = pd.DataFrame(eforest_stack_clf.predict_proba(xval_stack)[:, 1])
forest_stack_pred = pd.concat([eforest_stack_pred, rforest_stack_pred], axis=1)

stack_clf = LogisticRegression()
stack_clf.fit(forest_stack_pred, yval_stack)
stack_x = pd.concat([pd.DataFrame(rforest_pred), pd.DataFrame(eforest_pred)], axis=1)
stack_test_x = pd.concat([pd.DataFrame(rforest_test_pred), pd.DataFrame(eforest_test_pred)], axis=1)
stack_pred = stack_clf.predict_proba(stack_x)[:, 1]
# comparison = pd.concat([pd.DataFrame(stack_pred), pd.DataFrame(rforest_pred), pd.DataFrame(eforest_pred)], axis=1)

print "Training Automatic Stack"
lr_gen = LogisticRegression()
stack_clf = StackedClassifier(clfs=[rforest_clf, eforest_clf], gen=lr_gen, folds=5)
stack_pred = stack_clf.predict_proba(xtrain)[:, 1]
print "Automatic Stack Pred:\n", stack_pred
input("stopping here...")

# evaluate performance metrics
from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix
clf = stack_clf # pick a classifier
clf_pred = stack_pred
cv_x = stack_x
fpr, tpr, thresholds = roc_curve(ytrain, clf_pred)
roc_auc = auc(fpr, tpr)

# model metrics and validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(clf, cv_x, ytrain, cv=8, verbose=150, scoring="roc_auc")
print "confusion_matrix:\n", confusion_matrix(ytrain, clf.predict(cv_x))
print "stack training set score: ", roc_auc_score(ytrain, clf_pred)
print "cross validation scores:\n", cv_scores
print "cv stats (mean, std): (%f, %f)" % (cv_scores.mean(), cv_scores.std())

# plot the roc curve
custom.roc_plot(fpr, tpr)
plt.show()
plt.close()

# submit solution
xtest = stack_test_x # for stacking
submit = pd.DataFrame()
submit["id"] = test.index
submit["Y"] = clf.predict_proba(xtest)[:, 1]
submit.to_csv("stack.csv", index=False)
