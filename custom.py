# custom.py
# custom helper functions

import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import numpy as np

def count_plot(data, y):
    # plot frequencies
    ax = sns.countplot(data, hue=y)
    count, k, n = [], 0, len(ax.patches)
    for patch in ax.patches:
        height = patch.get_height()
        if k < (n / 2):
            count.append(height) # 0 count
        elif not np.isnan(height):
            text = "{:0.4f}".format(height / (count[k - (n / 2)] + height))
            # print height, count[k - (n / 2)]
            ax.text(patch.get_x(), height + 3, text)
        k += 1

def roc_plot(fpr, tpr):
    # plot roc curve
    from sklearn.metrics import auc
    plt.plot(fpr, tpr, color="r", label="ROC Curve (area = %0.2f)" % auc(fpr, tpr))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc="lower right")
    plt.title("Receiver Operating Curve")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.tight_layout()

def eda_countplot(data, y):
    for feat in data:
        print data[feat].describe()
        plt.title(feat)
        # sns.distplot(data[feat], kde=False)
        count_plot(data[feat].astype(int), y)
        plt.show()
        plt.close()

def get_interactions(data, feats=None):
    interactionFeatures = set()
    copy = data
    if not feats:
        feats = data.columns
    for feat1 in feats:
        for feat2 in data.columns:
            if (feat1 > feat2):
                interactionFeatures.add((feat1, feat2))
            elif (feat2 > feat1):
                interactionFeatures.add((feat2, feat1))

    for (feat1, feat2) in interactionFeatures:
        copy[feat1 + "x" + feat2] = copy[feat1] * copy[feat2]

    return copy


# Stacked Classifier
# Implements a two level wolpert stacked classifier.
# level1 generalization is trained off generalizations from level 0
# refer to dr. wolpert's famous 1992 paper on stacked generalizations.
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
class StackedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clfs, gen, folds=5):
        self.skf = StratifiedKFold(n_splits=folds, shuffle=True)
        self.clfs = clfs # stacked classifiers
        self.gen = gen # level1 generalizer
        self.folds = folds

    def fit(self, X, y):
        ytest = pd.DataFrame()
        level1 = pd.DataFrame()
        for train_index, test_index in self.skf.split(X, y):
            # print train_index, "\n", test_index
            xtrain, xtest = X.iloc[train_index, :], X.iloc[test_index, :]
            ytrain = y.iloc[train_index]
            ytest = pd.concat([ytest, pd.Series(y.iloc[test_index])], axis=0, ignore_index=True)
            
            # train level 0
            level0 = pd.DataFrame()
            for clf in self.clfs:
                clf.fit(xtrain, ytrain)
                clf_pred = pd.DataFrame(clf.predict_proba(xtest)[:, 1])
                level0 = pd.concat([level0, clf_pred], axis=1, ignore_index=True)

            level1 = pd.concat([level1, level0], axis=0, ignore_index=True)

        # print "level1:", level1.shape, "\n", level1.head()
        self.gen.fit(level1, ytest.values.ravel())
        return self

    def predict_proba(self, X):
        # predict level 0
        level1 = pd.DataFrame()
        for clf in self.clfs:
            clf_pred = pd.DataFrame(clf.predict_proba(X)[:, 1])
            level1 = pd.concat([level1, clf_pred], axis=1, ignore_index=True)

        return self.gen.predict_proba(level1)

    def predict(self, X):
        # predict level 0
        level1 = pd.DataFrame()
        for clf in self.clfs:
            clf_pred = pd.DataFrame(clf.predict_proba(X)[:, 1])
            level1 = pd.concat([level1, clf_pred], axis=1, ignore_index=True)

        return self.gen.predict(level1)

# Extra Stacked Classifier
# Implements a two level wolpert stacked classifier.
# level1 generalization is trained off generalizations from level 0
# refer to dr. wolpert's famous 1992 paper on stacked generalizations.
# adds original features in next layer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
class ExtraStackedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clfs, gen, folds=5):
        self.skf = StratifiedKFold(n_splits=folds, shuffle=True)
        self.clfs = clfs # stacked classifiers
        self.gen = gen # level1 generalizer
        self.folds = folds

    def fit(self, X, y):
        xtest = pd.DataFrame()
        ytest = pd.DataFrame()
        level1 = pd.DataFrame()
        for train_index, test_index in self.skf.split(X, y):
            # print train_index, "\n", test_index
            xtrain = X.iloc[train_index, :] 
            ytrain = y.iloc[train_index]
            ytest = pd.concat([ytest, pd.Series(y.iloc[test_index])], axis=0, ignore_index=True)
            xtest = pd.concat([xtest, pd.DataFrame(X.iloc[test_index, :])], axis=0, ignore_index=True)
            
            # train level 0
            level0 = pd.DataFrame()
            for clf in self.clfs:
                clf.fit(xtrain, ytrain)
                clf_pred = pd.DataFrame(clf.predict_proba(X.iloc[test_index, :])[:, 1])
                level0 = pd.concat([level0, clf_pred], axis=1, ignore_index=True)

            level1 = pd.concat([level1, level0], axis=0, ignore_index=True)

        print "level1:", level1.shape, "\n", level1.head()
        self.gen.fit(pd.concat([xtest, level1], axis=1), ytest)
        return self

    def predict_proba(self, X):
        # predict level 0
        level1 = pd.DataFrame()
        for clf in self.clfs:
            clf_pred = pd.Series(clf.predict_proba(X)[:, 1])
            level1 = pd.concat([level1, clf_pred], axis=1, ignore_index=True)
        
        xpredict = pd.concat([
            pd.DataFrame(X.reset_index(drop=True)), pd.DataFrame(level1)], 
            axis=1, ignore_index=True)
        return self.gen.predict_proba(xpredict)

