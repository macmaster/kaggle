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
data.drop(["F9", "F16", "F21", "F5", "F6"], axis=1, inplace=True) # round 2 forest selection

# data.drop(["F19", "F11", "F10"], axis=1, inplace=True) # round 3 extra forest selection
# data.drop(["F11"], axis=1, inplace=True) # round 3 extra forest selection
### data.drop(["F22", "F27"], axis=1, inplace=True) # round 4 extra extra forest selection (too much)

# interaction features
# data = custom.get_interactions(data, ["F3", "F27", "F18", "F19", "F14"])

# unskew some features with boxcox
from scipy.stats import skew, boxcox
from sklearn.preprocessing import scale, robust_scale, minmax_scale

def data_scaling(X):
    data = pd.DataFrame(X)
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

    data.fillna(data.median(), inplace=True)
    return data

def ordinary_data_scaling(X):
    data = pd.DataFrame(X)
    for feat in data:
        data[feat] = scale(data[feat])
    return data

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
    n_estimators = 150, criterion="entropy", 
    min_samples_leaf=92, max_leaf_nodes=300,
    oob_score=True, max_features="log2",)
    # class_weight={1 : 2.75})

# train an extra random forest
# best model params(800):  {'max_features': 'log2', 'max_leaf_nodes': 400, 'criterion': 'entropy', 'min_samples_leaf': 12}
from sklearn.ensemble import ExtraTreesClassifier
eforest_clf = ExtraTreesClassifier( n_jobs = 2,
    n_estimators = 60, criterion="entropy",
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

###### Forest Stack
# run a stacking generalization over the 2 forest models
from sklearn.linear_model import LogisticRegression
from custom import StackedClassifier, ExtraStackedClassifier
stack_gen = LogisticRegression()
stack_clf = StackedClassifier(clfs=[xgb_clf, eforest_clf, gaussian_clf], gen=stack_gen, folds=5)

# Resample the dataset to better represent minority
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTETomek
xcols = xtrain.columns
xorig, yorig = xtrain, ytrain
xtrain, xval, ytrain, yval = \
    train_test_split(xorig, yorig, test_size=0.2)

from imblearn.pipeline import make_pipeline
# resamp = RandomUnderSampler(ratio=0.5)
resamp = make_pipeline(TomekLinks())
# resamp = SMOTETomek(smote=SMOTE(ratio=0.075, kind="borderline2"))
xtrain, ytrain = resamp.fit_sample(xtrain, ytrain)
xtrain = pd.DataFrame(xtrain, columns=xcols)
print "xtrain after resample and split: ", xtrain.shape
print "y count:\n", pd.Series(ytrain).value_counts()
sns.distplot(ytrain, kde=False)
plt.title("Resampled Y")
plt.show()
plt.close()

# # optional data scaling
xtrain = data_scaling(xtrain)
xtest = data_scaling(xtest)
xval = data_scaling(xval)
# xval = ordinary_data_scaling(xval)
# xtrain = ordinary_data_scaling(xtrain)


rforest_clf.fit(xtrain, ytrain)
importances = pd.Series(rforest_clf.feature_importances_, index=xtrain.columns.values)
print "Random Forest Feature Importances:\n", importances.sort_values()
print "Random Forest oob score: ", rforest_clf.oob_score_

# evaluate performance metrics
from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix
clf = rforest_clf # pick a classifier
print "classifier:", clf

# optional EDA
# custom.eda_scatter(xtrain, ytrain, ["F3", "F14", "F25", "F2", "F18"])
# custom.eda_interactions(xtrain, ytrain, ["F14", "F25", "F2"])
custom.eda_scatter(xtrain, ytrain, ["unskewedF3", "unskewedF14", "unskewedF25", "unskewedF2", "scaledF18"])
# scaledF18      0.057038
# unskewedF2     0.111228
# unskewedF25    0.157828
# unskewedF14    0.183419
# unskewedF3     0.290290

# custom.eda_countplot(xtrain, ytrain)

cv_scores = custom.resamp_cross_val_score(resamp, clf, xorig, yorig, cv=5, verbose=True)
print "cross validation scores:\n", cv_scores
print "cv stats(mean, std): (%f, %f)" % (np.mean(cv_scores), np.std(cv_scores))

# plot the roc curve
clf.fit(xtrain, ytrain)
clf_pred = clf.predict_proba(xval)[:, 1]
fpr, tpr, thresholds = roc_curve(yval, clf_pred)
roc_auc = auc(fpr, tpr)
custom.roc_plot(fpr, tpr)

# model metrics and validation
print "confusion_matrix:\n", confusion_matrix(yval, clf.predict(xval))
print "stack training set score: ", roc_auc_score(yval, clf_pred)

plt.show()
plt.close()

# submit solution
submit = pd.DataFrame()
submit["id"] = test.index
submit["Y"] = clf.predict_proba(xtest)[:, 1]
submit.to_csv("smote.csv", index=False)
