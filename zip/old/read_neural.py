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

# drop some collinear features
data.drop(["F26"], axis=1, inplace=True)

# junk features
data.drop(["F4", "F7", "F8", "F15", "F17", "F20", "F24"], axis=1, inplace=True)
data.drop(["F1", "F12", "F13"], axis=1, inplace=True)
data.drop(["F9", "F16", "F21"], axis=1, inplace=True) 

# scale some features with boxcox 
from scipy.stats import skew, boxcox
from sklearn.preprocessing import scale, robust_scale, minmax_scale
data.fillna(data.mean(), inplace=True)
data = data.apply(scale)

print "data post processing: ", data.shape
xtest = data[train.shape[0]:]
xtrain = data[:train.shape[0]]
ytrain = train["Y"]

# optional EDA
# custom.eda_countplot(xtrain, ytrain)
# custom.eda_heatmap(xtrain)
# custom.eda_boxplot(xtrain, ytrain)# train a neural network

# multi layer perceptron
from sklearn.neural_network import MLPClassifier
mlp_clf = MLPClassifier(hidden_layer_sizes=(40,50), alpha=0.01)

# train a keras neural network
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_auc_score
def create_nn():
    model = Sequential()
    model.add(Dense(14, input_dim=xtrain.shape[1], activation="relu"))
    model.add(Dense(7, activation="relu"))
    model.add(Dense(1, activation="softmax"))
    model.compile( # binary classification
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"])
    return model

keras_clf = KerasClassifier(
    build_fn=create_nn, epochs=150, 
    batch_size=32, verbose=0)

# evaluate performance metrics
from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix
clf = keras_clf # pick a classifier
clf.fit(xtrain.values, ytrain.values)
clf_pred = clf.predict_proba(xtrain.values)[:, 1]
fpr, tpr, thresholds = roc_curve(ytrain.values, clf_pred)
roc_auc = auc(fpr, tpr)
print "classifier:", clf

# model metrics and validation
from sklearn.model_selection import cross_val_score, StratifiedKFold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cv_scores = cross_val_score(clf, xtrain.values, ytrain.values, cv=kfold, verbose=150, scoring="roc_auc")
print "confusion_matrix:\n",confusion_matrix(ytrain.values, clf.predict(xtrain.values))
print "stack training set score: ", roc_auc_score(ytrain.values, clf_pred)
print "cross validation scores:\n", cv_scores
print "cv stats(mean, std): (%f, %f)" % (cv_scores.mean(), cv_scores.std())

# save neural net to disk
from keras.models import model_from_json
model = keras_clf.model
with open("neural.json", "w") as json:
    json.write(model.to_json())
model.save_weights("neural.h5")

# plot the roc curve
# custom.roc_plot(fpr, tpr)
# plt.show()
# plt.close()

# submit solution
submit = pd.DataFrame()
submit["id"] = test.index
submit["Y"] = clf.predict_proba(xtest.values)[:, 1]
submit.to_csv("stack.csv", index=False)


# from sklearn.model_selection import GridSearchCV
# mlp_grid = {
#     "hidden_layer_sizes" : [
#         (40, 40), (40, 50), (50, 50),
#         (50, 40), (45, 45),
#     ],
# 
#     "alpha" : [0.0001, 0.01, 1.0, 10.0],
# }
# # grid_search = GridSearchCV(neural_clf, neural_grid, cv=4, verbose=1500, scoring="roc_auc")
# # grid_search.fit(xtrain, ytrain)
# # print "best model params: ", grid_search.best_params_
# # print "best model cv score: ", grid_search.best_score_
