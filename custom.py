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

def eda_scatter(data, y, feats, clf=None):
    feats = set([(min(feat1, feat2), max(feat1, feat2)) 
        for feat1 in feats for feat2 in feats])
    for (feat1, feat2) in feats:
        if clf:
            x_min, x_max = data[feat1].min() - 1, data[feat1].max() + 1
            y_min, y_max = data[feat2].min() - 1, data[feat2].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
            zz = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            zz = zz.reshape(xx.shape)
            plt.contourf(xx, yy, z, alpha=0.4)
        plt.scatter(data[feat1], data[feat2], c=y, s=20, edgecolor='k')
        plt.title("%s vs %s scatter" % (feat1, feat2))
        plt.tight_layout()
        plt.show()
        plt.close()

def eda_interactions(data, y, feats):
    feats = set([(min(feat1, feat2), max(feat1, feat2)) 
        for feat1 in feats for feat2 in feats])
    for (feat1, feat2) in feats:
        f1xf2 = data[feat1] * data[feat2] * data[feat2]
        plt.scatter(data[feat1], f1xf2, c=y, alpha=0.4)
        plt.title("%s vs %s scatter" % (feat1, feat2))
        plt.tight_layout()
        plt.show()
        plt.close()

def get_interactions(data, feats=None):
    copy = data
    if not feats: feats = data.columns
    feats = set([(min(feat1, feat2), max(feat1, feat2)) 
        for feat1 in feats for feat2 in feats])

    for (feat1, feat2) in feats:
        copy[feat1 + "x" + feat2] = copy[feat1] * copy[feat2]

    return copy

def set_interactions(data, feats):
    copy = data
    for (feat1, feat2) in feats:
        copy[feat1 + "x" + feat2] = copy[feat1] * copy[feat2]
    return copy

def resamp_cross_val_score(resamp, clf, xtrain, ytrain, cv=5, verbose=False):
    cv_scores, k = [], 1
    xcols = xtrain.columns
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    skf = StratifiedKFold(n_splits=cv, shuffle=True)
    for train_index, test_index in skf.split(xtrain, ytrain):
        ## resample and balance distribution
        xres, yres = (xtrain.iloc[train_index, :], ytrain.iloc[train_index])
        xres, yres = resamp.fit_sample(xres, yres)
        xres = pd.DataFrame(xres, columns=xcols)

        clf.fit(xres, yres)
        clf_pred = clf.predict_proba(xtrain.iloc[test_index, :])[:, 1]
        score = roc_auc_score(ytrain.iloc[test_index], clf_pred)
        
        if verbose: print "[CV] fold (%d) auc: (%f)" % (k, score)
        cv_scores.append(score)
        k += 1
    return cv_scores

from sklearn.base import BaseEstimator, TransformerMixin
class PandasColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feats):
        self.feats = feats

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.feats]

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
        self.level1 = level1
        return self

    def predict_proba(self, X):
        # predict level 0
        level1 = pd.DataFrame()
        for clf in self.clfs:
            clf_pred = pd.DataFrame(clf.predict_proba(X)[:, 1])
            level1 = pd.concat([level1, clf_pred], axis=1, ignore_index=True)
        
        self.level1 = level1
        return self.gen.predict_proba(level1)

    def predict(self, X):
        # predict level 0
        level1 = pd.DataFrame()
        for clf in self.clfs:
            clf_pred = pd.DataFrame(clf.predict_proba(X)[:, 1])
            level1 = pd.concat([level1, clf_pred], axis=1, ignore_index=True)

        self.level1 = level1
        return self.gen.predict(level1)
    
    def grid_search(self, gen_grid, X, y):
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
            
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(self.gen, gen_grid, cv=5, verbose=1500, scoring="roc_auc")
        grid_search.fit(level1, ytest.values.ravel())
        return grid_search

    def heatmap(self):
        level1 = pd.DataFrame(self.level1)
        ax = sns.heatmap(level1.corr(), vmin=-1.0, vmax=1.0, annot=True, fmt=".2f")
        ax.set_title("Stacking Correlation Heatmap (level1)")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.tight_layout()
        plt.show()
        plt.close()


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

