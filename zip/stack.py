# stack.py
# model for the kaggle competition
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

# drop some collinear features
data.drop(["F26"], axis=1, inplace=True)

# junk features
data.drop(["F4", "F7", "F8", "F15", "F17", "F20", "F24"], axis=1, inplace=True)
data.drop(["F1", "F12", "F13"], axis=1, inplace=True) # further random forest selection
data.drop(["F9", "F21"], axis=1, inplace=True) # round 2 forest selection
# data.drop(["F9", "F16", "F21"], axis=1, inplace=True) # round 2 forest selection

# scale some features with boxcox 
from scipy.stats import skew, boxcox
from sklearn.preprocessing import scale, robust_scale, minmax_scale
data.fillna(data.mean(), inplace=True)

# data transformations
data["F6"] = np.log(data["F6"])
data["F16"] = np.log(data["F16"])
data = data.apply(scale)

print "data post processing: ", data.shape
print "features:", data.columns
xtest = data[train.shape[0]:]
xtrain = data[:train.shape[0]]
ytrain = train["Y"]

# optional EDA
# custom.eda_countplot(xtrain, ytrain)
# custom.eda_heatmap(xtrain)
# custom.eda_boxplot(xtrain, ytrain)

# train a random forest 
# best model params(1800):  {'max_features': 'log2', 'max_leaf_nodes': 300, 'criterion': 'entropy', 'min_samples_leaf': 92}
from sklearn.ensemble import RandomForestClassifier
rforest_clf = RandomForestClassifier( n_jobs = 2,
    n_estimators = 160, criterion="entropy", 
    min_samples_leaf=90, max_leaf_nodes=325,
    oob_score=True, max_features="log2")

# train an extra random forest
# best model params(800):  {'max_features': 'log2', 'max_leaf_nodes': 400, 'criterion': 'entropy', 'min_samples_leaf': 12}
from sklearn.ensemble import ExtraTreesClassifier
eforest_clf = ExtraTreesClassifier( n_jobs = 2,
    n_estimators = 160, criterion="entropy",
    min_samples_leaf=8, max_leaf_nodes=560,
    # min_samples_leaf=3, max_leaf_nodes=425, # interaction features
    max_features="log2")

eforest_grid = {
    "min_samples_leaf" : [8],
    # "max_leaf_nodes" : [310, 325, 340],
    "max_leaf_nodes" : [540, 560, 580],
}

# from sklearn.model_selection import GridSearchCV
# grid_search = GridSearchCV(eforest_clf, eforest_grid, cv=7, verbose=5000, scoring="roc_auc")
# grid_search.fit(xtrain, ytrain)
# print "best model params: ", grid_search.best_params_
# print "best model cv score: ", grid_search.best_score_

# train a neural network
from sklearn.neural_network import MLPClassifier
neural_clf = MLPClassifier()
# best gen model params:  {'alpha': 0.01, 'activation': 'relu', 'hidden_layer_sizes': (36,)}

neural_grid = {
    "hidden_layer_sizes" : [(36,)],
    "activation" : ["logistic", "relu"],
    "alpha" : [0.01, 0.1, 0.5],
    # "batch_size" : [70, 80, 90]
}
# from sklearn.model_selection import GridSearchCV
# grid_search = GridSearchCV(neural_clf, neural_grid, cv=7, verbose=5000, scoring="roc_auc")
# grid_search = grid_search.fit(xtrain, ytrain)
# print "best gen model params: ", grid_search.best_params_
# print "best gen model cv score: ", grid_search.best_score_

# train a native bayes classifier
from sklearn.naive_bayes import GaussianNB
gaussian_clf = GaussianNB()

# train a K nearest neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_jobs=2, n_neighbors=300)

# train an xgboost model
# {'colsample_bytree': 0.7, 'scale_pos_weight': 1, 'learning_rate': 0.05, 'min_child_weight': 3, 'n_estimators': 160, 'subsample': 0.7, 'max_depth': 4, 'gamma': 0 
import xgboost as xgb
import random
xgb_clf = xgb.XGBClassifier( nthread = 2,
    n_estimators = 515, max_depth=4,
    learning_rate=0.02, gamma=0.37,
    min_child_weight=2.4, scale_pos_weight=1,
    subsample=0.76, colsample_bytree=0.58,
    reg_alpha=4.5, seed=random.randint(0, 50),    
#     n_estimators = 515, max_depth=4,
#     learning_rate=0.02, gamma=0.37,
#     min_child_weight=2.35, scale_pos_weight=0.95,
#     subsample=0.72, colsample_bytree=0.58,
#     reg_alpha=4.5, seed=random.randint(0, 50),
)
xgb1_clf = xgb.XGBClassifier( nthread = 2,
    n_estimators = 915, max_depth=3,
    learning_rate=0.02, gamma=1.2,
    min_child_weight=2, scale_pos_weight=0.9,
    subsample=0.65, colsample_bytree=0.6,
    reg_alpha=4.5, seed=random.randint(0, 50),
#     n_estimators = 915, max_depth=3,
#     learning_rate=0.02, gamma=1.05,
#     min_child_weight=2, scale_pos_weight=1,
#     subsample=0.6, colsample_bytree=0.6,
#     reg_alpha=1, seed=random.randint(0, 50),
)
xgb2_clf = xgb.XGBClassifier( nthread = 2,
    n_estimators = 550, max_depth=5,
    learning_rate=0.02, gamma=1.3,
    min_child_weight=2, scale_pos_weight=1,
    subsample=0.66, colsample_bytree=0.65,
    reg_alpha=7.25, seed=random.randint(0, 50),
#     n_estimators = 450, max_depth=5,
#     learning_rate=0.03, gamma=0.9,
#     min_child_weight=2.2, scale_pos_weight=1,
#     subsample=0.66, colsample_bytree=0.6,
#     reg_alpha=2.4, seed=random.randint(0, 50),
)

xgb_params = {
    "learning_rate" : 0.02,
    "max_depth" : 5,
    "gamma" : 1.3,
    "subsample" : 0.66,
    "colsample_bytree" : 0.65,
    "min_child_weight" : 2,
    "scale_pos_weight" : 1,
    "reg_alpha" : 7.25
}

# # built in cv for n estimators
# xgtrain = xgb.DMatrix(xtrain.values, label=ytrain.values)
# cvresult = xgb.cv(xgb_params, xgtrain, verbose_eval=False, 
#     num_boost_round=3000, nfold=5, stratified=True, metrics="auc",
#     early_stopping_rounds=100,)
# print "xgb.cv result: ", cvresult.sort_values(by="test-auc-mean")

from sklearn.model_selection import GridSearchCV
xgb_grid = {
    "n_estimators" : [500, 550, 600], # 1200 for 0.01
    "learning_rate" : [0.02, 0.03],
    "max_depth" : [5],
    "gamma" : [1.3],
    "min_child_weight" : [2],
    "scale_pos_weight" : [1],
    "subsample" : [0.66],
    "colsample_bytree" : [0.65],
    "reg_alpha" : [7.25], # 1.1
}

# grid_search = GridSearchCV(xgb2_clf, xgb_grid, cv=12, verbose=5000, scoring="roc_auc")
# grid_search.fit(xtrain, ytrain)
# print "best model params: ", grid_search.best_params_
# print "best model cv score: ", grid_search.best_score_

# rforest_clf.fit(xtrain, ytrain)
# importances = pd.Series(rforest_clf.feature_importances_, index=xtrain.columns.values)
# print "Random Forest Feature Importances:\n", importances.sort_values()
# print "Random Forest oob score: ", rforest_clf.oob_score_

# eforest_clf.fit(xtrain, ytrain)
# importances = pd.Series(eforest_clf.feature_importances_, index=xtrain.columns.values)
# print "Extra Random Forest Feature Importances:\n", importances.sort_values()


# # # # plot xgb feature importance
# # # # xgb feature selection
# xgb_clf = xgb_clf.fit(xtrain, ytrain, eval_metric="auc")
# importances = pd.Series(xgb_clf.feature_importances_, index=xtrain.columns.values)
# print "XGB Feature Importances:\n", importances.sort_values()
# xgb.plot_importance(xgb_clf)
# plt.title("XGB Feature Importance")
# plt.show(); plt.close()

###### Forest Stack
# run a stacking generalization over the 2 forest models
from sklearn.linear_model import LogisticRegression
from custom import StackedClassifier, ExtraStackedClassifier
# stack_gen = LogisticRegression(penalty="l2")

# old from tuesday.
# best gen model params:  {'alpha': 1, 'activation': 'logistic', 'batch_size': 80, 'hidden_layer_sizes': (110, 110)}
# best gen model cv score:  0.862517037271

# generalizer 1: level1 cv error:
stack1_gen = MLPClassifier(hidden_layer_sizes=(36,18), 
    activation="relu", alpha=0.02, batch_size=100)

# # generalizer 2: level1 cv error: 0.8624
stack2_gen = xgb.XGBClassifier( nthread = 2,
    n_estimators = 215, max_depth=2,
    learning_rate=0.02, gamma=0.07,
    min_child_weight=0.25, scale_pos_weight=1.05,
    subsample=0.7, colsample_bytree=0.52,
    reg_alpha=0.02, seed=random.randint(0, 50),    
)

# stack_clf = StackedClassifier(clfs=[xgb_clf, eforest_clf, gaussian_clf], gen=stack_gen, folds=5) # simple cv optimal
# stack_clf = StackedClassifier(clfs=[xgb_clf, rforest_clf, eforest_clf], gen=stack_gen, folds=7) # option 2 cv optimal
# stack_clf = StackedClassifier(clfs=[xgb2_clf, rforest_clf, eforest_clf], gen=stack1_gen, folds=7) # option 1 cv optimal
stack_clf = StackedClassifier(clfs=[xgb_clf, xgb1_clf, xgb2_clf], gen=stack1_gen, folds=7) # option 1 cv optimal

# # generate level1 training set. 
# level1x, level1y = stack_clf.level1_set(xtrain, ytrain)
# level1 = pd.DataFrame(level1x)
# level1["y"] = level1y 
# level1.to_csv("level1.csv")

# from mlxtend.classifier import StackingClassifier
# stack_clf = StackingClassifier(
#     classifiers=[xgb_clf, rforest_clf, eforest_clf], 
#     meta_classifier=stack_gen,
#     use_probas=True, average_probas=False,
# )

from sklearn.model_selection import GridSearchCV
stack_grid = {
    "clfs" : [
        [xgb_clf, xgb1_clf, xgb2_clf],
        [xgb_clf, rforest_clf, eforest_clf], # works extremely well with neural net generalizer
        # [xgb1_clf, xgb2_clf, eforest_clf],
        # [xgb1_clf, xgb2_clf, rforest_clf],
        # [xgb1_clf, rforest_clf, eforest_clf],
        [xgb2_clf, rforest_clf, eforest_clf], 
    ],
    "gen" : [stack1_gen, stack2_gen],
    "folds" : [7],
}
# grid_search = GridSearchCV(stack_clf, stack_grid, cv=7, verbose=1500, scoring="roc_auc")
# grid_search.fit(xtrain, ytrain)
# print "best stack model params: ", grid_search.best_params_
# print "best stack model cv score: ", grid_search.best_score_

# evaluate performance metrics
from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix
clf = xgb2_clf # pick a classifier
clf.fit(xtrain, ytrain)
clf_pred = clf.predict_proba(xtrain)[:, 1]
fpr, tpr, thresholds = roc_curve(ytrain, clf_pred)
roc_auc = auc(fpr, tpr)
print "classifier:", clf

# # model metrics and validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(clf, xtrain, ytrain, cv=12, verbose=150, scoring="roc_auc")
# print "confusion_matrix:\n",confusion_matrix(ytrain, clf.predict(xtrain))
# print "stack training set score: ", roc_auc_score(ytrain, clf_pred)
print "cross validation scores:\n", cv_scores
print "cv stats(mean, std): (%f, %f)" % (cv_scores.mean(), cv_scores.std())

# submit solution
submit = pd.DataFrame()
submit["id"] = test.index
submit["Y"] = clf.predict_proba(xtest)[:, 1]
submit.to_csv("stack.csv", index=False)
