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
import random

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

seed = random.randint(0, 50)
mpl.style.use(["fivethirtyeight"])
sns.set(style="whitegrid", color_codes=True)

train = pd.read_csv("level1.csv", index_col=0)
xtrain = train.drop(["y"], axis=1)
ytrain = train["y"]

print train.head()

# optional EDA
# custom.eda_countplot(xtrain, ytrain)
# custom.eda_heatmap(xtrain)
# custom.eda_boxplot(xtrain, ytrain)# train a neural network

# multi layer perceptron
from sklearn.neural_network import MLPClassifier
mlp_clf = MLPClassifier(hidden_layer_sizes=(36,18), activation="relu", alpha=0.02, batch_size=100)

# train a keras neural network
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_auc_score
def create_nn():
    model = Sequential()
    model.add(Dense(3, input_dim=xtrain.shape[1], activation="relu"))
    model.add(Dense(3, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile( # binary classification
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"])
    return model

def create_nn2():
    model = Sequential()
    model.add(Dense(4, input_dim=xtrain.shape[1], activation="relu"))
    model.add(Dense(6, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile( # binary classification
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"])
    return model

def create_nn3():
    model = Sequential()
    model.add(Dense(6, input_dim=xtrain.shape[1], activation="relu"))
    model.add(Dense(3, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile( # binary classification
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"])
    return model

keras_clf = KerasClassifier(
    build_fn=create_nn, epochs=800, 
    batch_size=500, verbose=0)

keras2_clf = KerasClassifier(
    build_fn=create_nn, epochs=800, 
    batch_size=500, verbose=0)

keras3_clf = KerasClassifier(
    build_fn=create_nn, epochs=800, 
    batch_size=500, verbose=0)

# logistic regression generalizer
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(penalty="l2", C=0.015)

# # neural net takes friggn forever to train. 
# # use a subset of the data.
# from sklearn.model_selection import train_test_split
# xtrain, xval, ytrain, yval = \
#     train_test_split(xtrain, ytrain, stratify=ytrain, test_size=0.80)



from sklearn.model_selection import GridSearchCV
keras_grid = {
    "epochs" : [600, 800, 1200],
    "batch_size" : [32, 200, 500],
}
grid_search = GridSearchCV(keras_clf, keras_grid, cv=5, verbose=0, scoring="roc_auc")
grid_search.fit(xtrain.values, ytrain.values)
print "best model params: ", grid_search.best_params_
print "best model cv score: ", grid_search.best_score_

grid_search = GridSearchCV(keras2_clf, keras_grid, cv=5, verbose=0, scoring="roc_auc")
grid_search.fit(xtrain.values, ytrain.values)
print "best model params: ", grid_search.best_params_
print "best model cv score: ", grid_search.best_score_

grid_search = GridSearchCV(keras3_clf, keras_grid, cv=5, verbose=0, scoring="roc_auc")
grid_search.fit(xtrain.values, ytrain.values)
print "best model params: ", grid_search.best_params_
print "best model cv score: ", grid_search.best_score_

# gradient boosting machine.
import xgboost as xgb
xgb_clf = xgb.XGBClassifier( nthread = 2,
    n_estimators = 215, max_depth=2,
    learning_rate=0.02, gamma=0.07,
    min_child_weight=0.25, scale_pos_weight=1.05,
    subsample=0.7, colsample_bytree=0.52,
    reg_alpha=0.02, seed=random.randint(0, 50),    
)

xgb_params = {
    "learning_rate" : 0.02,
    "max_depth" : 2,
    "gamma" : 0.07,
    "subsample" : 0.65,
    "colsample_bytree" : 0.52,
    "min_child_weight" : 0.25,
    "scale_pos_weight" : 1.15,
    "reg_alpha" : 0.02
}

# # built in cv for n estimators
# xgtrain = xgb.DMatrix(xtrain.values, label=ytrain.values)
# cvresult = xgb.cv(xgb_params, xgtrain, verbose_eval=False, 
#     num_boost_round=1500, nfold=8, stratified=True, metrics="auc",
#     early_stopping_rounds=100,)
# print "xgb.cv result: ", cvresult.sort_values(by="test-auc-mean")
# input ("finished xgb cv")

from sklearn.model_selection import GridSearchCV
xgb_grid = {
    "n_estimators" : [215], # 1200 for 0.01
    "learning_rate" : [0.02],
    "max_depth" : [2],
    "gamma" : [0.07],
    "min_child_weight" : [0.25],
    "scale_pos_weight" : [1.05],
    "subsample" : [0.7],
    "colsample_bytree" : [0.52],
    "reg_alpha" : [0.02], # 1.1
}

# grid_search = GridSearchCV(xgb_clf, xgb_grid, cv=10, verbose=5000, scoring="roc_auc")
# grid_search.fit(xtrain, ytrain)
# print "best model params: ", grid_search.best_params_
# print "best model cv score: ", grid_search.best_score_



from sklearn.model_selection import GridSearchCV
lr_grid = {
    "C" : [0.001, 0.01, 0.1, 0.5, 1],
    "penalty" : ["l2", "l1"]
}

mlp_grid = {
    "hidden_layer_sizes" : [(36, 18)],
    "activation" : ["relu"],
    "alpha" : [0.02],
    "batch_size" : [100]
}

# grid_search = GridSearchCV(mlp_clf, mlp_grid, cv=10, verbose=1500, scoring="roc_auc")
# grid_search.fit(xtrain.values, ytrain.values)
# print "best model params: ", grid_search.best_params_
# print "best model cv score: ", grid_search.best_score_

# evaluate performance metrics
from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix
clf = keras_clf # pick a classifier
# clf.fit(xtrain.values, ytrain.values)
# clf_pred = clf.predict_proba(xtrain.values)[:, 1]
# fpr, tpr, thresholds = roc_curve(ytrain.values, clf_pred)
# roc_auc = auc(fpr, tpr)
# print "classifier:", clf

# model metrics and validation
from sklearn.model_selection import cross_val_score, StratifiedKFold
kfold = StratifiedKFold(n_splits=7, shuffle=True, random_state=seed)
cv_scores = cross_val_score(clf, xtrain.values, ytrain.values, cv=kfold, verbose=150, scoring="roc_auc")
# print "confusion_matrix:\n",confusion_matrix(ytrain.values, clf.predict(xtrain.values))
# print "stack training set score: ", roc_auc_score(ytrain.values, clf_pred)
print "cross validation scores:\n", cv_scores
print "cv stats(mean, std): (%f, %f)" % (cv_scores.mean(), cv_scores.std())

# clf.fit(xtrain.values, ytrain.values)
# clf_pred = clf.predict_proba(xval.values)[:, 1]
# print "validation score: ", roc_auc_score(yval, clf_pred)
# 
# plot the roc curve
# custom.roc_plot(fpr, tpr)
# plt.show()
# plt.close()
