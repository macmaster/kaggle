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

