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

# data.drop(["F19", "F11", "F6", "F10"], axis=1, inplace=True) # round 3 extra forest selection
data.drop(["F11", "F6"], axis=1, inplace=True) # round 3 extra forest selection
### data.drop(["F22", "F27"], axis=1, inplace=True) # round 4 extra extra forest selection (too much)

# interaction features
# data = custom.get_interactions(data, ["F3", "F27", "F18", "F19", "F14"])

# unskew some features with boxcox
from scipy.stats import skew, boxcox
from sklearn.preprocessing import scale, robust_scale, minmax_scale

skewedFeats = [
    "F2", "F14", "F25",             # hi numerical categorical features
    "F3", "F19", "F22", "F27",      # hi numerical quantitative features
    "F6",                           # features the trees like
]
robustFeats = ["F5", "F10"]
scaledFeats = list(set(data.columns) - set(skewedFeats) - set(robustFeats))

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
    n_estimators = 40, criterion="entropy", 
    min_samples_leaf=92, max_leaf_nodes=300,
    oob_score=True, max_features="log2")
rforest2_clf = RandomForestClassifier( n_jobs = 2,
    n_estimators = 40, criterion="gini", 
    min_samples_leaf=92, max_leaf_nodes=300,
    oob_score=True, max_features="log2")

# train an extra random forest
# best model params(800):  {'max_features': 'log2', 'max_leaf_nodes': 400, 'criterion': 'entropy', 'min_samples_leaf': 12}
from sklearn.ensemble import ExtraTreesClassifier
eforest_clf = ExtraTreesClassifier( n_jobs = 2,
    n_estimators = 60, criterion="entropy",
    min_samples_leaf=12, max_leaf_nodes=400,
    max_features="log2")
eforest2_clf = ExtraTreesClassifier( n_jobs = 2,
    n_estimators = 60, criterion="gini",
    min_samples_leaf=12, max_leaf_nodes=400,
    max_features="log2")

# train a native bayes classifier
from sklearn.naive_bayes import GaussianNB
gaussian_clf = GaussianNB()

# train a K nearest neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_jobs=2, n_neighbors=300)
# knn_clf = KNeighborsClassifier(n_jobs=2, n_neighbors=1)

# train an xgboost model
# {'colsample_bytree': 0.7, 'scale_pos_weight': 1, 'learning_rate': 0.05, 'min_child_weight': 3, 'n_estimators': 160, 'subsample': 0.7, 'max_depth': 4, 'gamma': 0 
import xgboost as xgb
import random
xgb_clf = xgb.XGBClassifier( nthread = 2,
    n_estimators = 550, max_depth=3,
    learning_rate=0.02, gamma=0.4,
    min_child_weight=2.5, scale_pos_weight=0.67,
    subsample=0.72, colsample_bytree=0.58,
    reg_alpha=2.5, seed=random.randint(0, 50),
)

# xgb_params = {
#     "n_estimators" : 3000,
#     "learning_rate" : 0.01,
#     "max_depth" : 4,
#     "subsample" : 0.72,
#     "colsample_bytree" : 0.58,
#     "min_child_weight" : 2.5,
#     "scale_pos_weight" : 1,
#     "gamma" : 0.35,
# }
# 
# # built in cv for n estimators
# xgtrain = xgb.DMatrix(xtrain.values, label=ytrain.values)
# cvresult = xgb.cv(xgb_params, xgtrain, verbose_eval=False, 
#     num_boost_round=1000, nfold=10, stratified=True, metrics="auc",
#     early_stopping_rounds=100,)
# print "xgb.cv result: ", cvresult.sort_values(by="test-auc-mean")
# input ("waiting to poll...")

from sklearn.model_selection import GridSearchCV
xgb_grid = {
    "n_estimators" : [550],
    "learning_rate" : [0.02],
    "max_depth" : [3],
    "gamma" : [0.40],
    "min_child_weight" : [2.5],
    "subsample" : [0.72],
    "colsample_bytree" : [0.58],
    "scale_pos_weight" : [0.65, 0.67, 0.69],
    "reg_alpha" : [2.5], # 1.1
}

grid_search = GridSearchCV(xgb_clf, xgb_grid, cv=5, verbose=5000, scoring="roc_auc")
grid_search.fit(xtrain, ytrain)
print "best model params: ", grid_search.best_params_
print "best model cv score: ", grid_search.best_score_

# # from sklearn.externals import joblib
# # joblib.dump(rforest_clf, "models/random_forest.pkl")
# # joblib.dump(eforest_clf, "models/extra_random_forest.pkl")
# rforest_clf.fit(xtrain, ytrain)
# importances = pd.Series(rforest_clf.feature_importances_, index=xtrain.columns.values)
# print "Random Forest Feature Importances:\n", importances.sort_values()
# print "Random Forest oob score: ", rforest_clf.oob_score_

# eforest_clf.fit(xtrain, ytrain)
# importances = pd.Series(eforest_clf.feature_importances_, index=xtrain.columns.values)
# print "Extra Random Forest Feature Importances:\n", importances.sort_values()
# # rforest_clf = joblib.load("models/random_forest.pkl")
# # eforest_clf = joblib.load("models/extra_random_forest.pkl")

# # plot xgb feature importance
# xgb_clf = xgb_clf.fit(xtrain, ytrain, eval_metric="auc")
# xgb.plot_importance(xgb_clf)
# plt.title("XGB Feature Importance")
# plt.show(); plt.close()

###### Forest Stack
# run a stacking generalization over the 2 forest models
from sklearn.linear_model import LogisticRegression
from custom import StackedClassifier, ExtraStackedClassifier
stack_gen = LogisticRegression()
# stack_gen = KNeighborsClassifier(n_jobs=2, n_neighbors=500)
# stack_clf1 = StackedClassifier(clfs=[rforest_clf, xgb_clf], gen=stack_gen, folds=5)
# stack_clf2 = StackedClassifier(clfs=[knn_clf, gaussian_clf], gen=stack_gen, folds=5)
stack_clf = StackedClassifier(clfs=[xgb_clf, eforest_clf, gaussian_clf], gen=stack_gen, folds=5)

# from mlxtend.classifier import StackingClassifier
# stack_clf = StackingClassifier(
#     classifiers=[stack_clf1, stack_clf2], 
#     meta_classifier=stack_gen,
#     use_probas=True, average_probas=False,
# )

from sklearn.model_selection import GridSearchCV
stack_grid = {
    "clfs" : [
        [xgb_clf, rforest_clf, eforest_clf],
        [xgb_clf, rforest_clf, rforest2_clf],
        [xgb_clf, eforest_clf, eforest2_clf],
        [xgb_clf, rforest_clf, knn_clf],
        [xgb_clf, eforest_clf, knn_clf],
        [xgb_clf, rforest_clf, gaussian_clf],
        [xgb_clf, eforest_clf, gaussian_clf], # best with LR
    ],
    "gen" : [LogisticRegression(), KNeighborsClassifier(n_jobs=2, n_neighbors=200)],
    "folds" : [4],
}

# grid_search = GridSearchCV(stack_clf, stack_grid, cv=4, verbose=1500, scoring="roc_auc")
# grid_search.fit(xtrain, ytrain)
# print "best model params: ", grid_search.best_params_
# print "best model cv score: ", grid_search.best_score_

# evaluate performance metrics
from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix
clf = xgb_clf # pick a classifier
clf.fit(xtrain, ytrain)
clf_pred = clf.predict_proba(xtrain)[:, 1]
fpr, tpr, thresholds = roc_curve(ytrain, clf_pred)
roc_auc = auc(fpr, tpr)
print "classifier:", clf

# model metrics and validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(clf, xtrain, ytrain, cv=5, verbose=150, scoring="roc_auc")
# print "confusion_matrix:\n",confusion_matrix(ytrain, clf.predict(xtrain))
print "stack training set score: ", roc_auc_score(ytrain, clf_pred)
print "cross validation scores:\n", cv_scores
print "cv stats(mean, std): (%f, %f)" % (cv_scores.mean(), cv_scores.std())

# plot the roc curve
custom.roc_plot(fpr, tpr)
plt.show()
plt.close()

# submit solution
submit = pd.DataFrame()
submit["id"] = test.index
submit["Y"] = clf.predict_proba(xtest)[:, 1]
submit.to_csv("stack.csv", index=False)
