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

data.drop(["F19", "F11", "F6", "F10"], axis=1, inplace=True) # round 3 extra forest selection
### data.drop(["F22", "F27"], axis=1, inplace=True) # round 4 extra extra forest selection (too much)

# unskew some features with boxcox
from scipy.stats import skew, boxcox
from sklearn.preprocessing import scale, robust_scale, minmax_scale

skewedFeats = [
    "F2", "F14", "F25",             # hi numerical categorical features
    "F3", "F19", "F22", "F27",      # hi numerical quantitative features
]
robustFeats = ["F5", "F10"]
scaledFeats = list(set(data.columns) - set(skewedFeats) - set(robustFeats))

for feat in data:
    if (feat in skewedFeats):
        data["unskewed" + feat] = scale(boxcox(data[feat] + 1)[0])
    if (feat in scaledFeats):
        data["scaled" + feat] = minmax_scale(data[feat])
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

# train a neural network
from sklearn.neural_network import MLPClassifier
neural_clf = MLPClassifier(hidden_layer_sizes=(40,50), alpha=0.01)

# optional EDA
# custom.eda_countplot(xtrain, ytrain)

from sklearn.model_selection import GridSearchCV
neural_grid = {
    "hidden_layer_sizes" : [
        (40, 40), (40, 50), (50, 50),
        (50, 40), (45, 45),
    ],

    "alpha" : [0.0001, 0.01, 1.0, 10.0],
}
# grid_search = GridSearchCV(neural_clf, neural_grid, cv=4, verbose=1500, scoring="roc_auc")
# grid_search.fit(xtrain, ytrain)
# print "best model params: ", grid_search.best_params_
# print "best model cv score: ", grid_search.best_score_

# evaluate performance metrics
from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix
clf = neural_clf # pick a classifier
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
submit.to_csv("neural.csv", index=False)
