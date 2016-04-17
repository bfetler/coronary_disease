# -*- coding: utf-8 -*-

# coronary artery (heart) disease prediction

import os
import pandas as pd
from pandas.tools.plotting import scatter_matrix as pd_scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression as lr

def get_plotdir():
    plotdir = "coronary_disease_plots/"
    return plotdir

def make_plotdir():
    sns.set_style("darkgrid")
    plotdir = get_plotdir()
    if not os.access(plotdir, os.F_OK):
        os.mkdir(plotdir)
    return plotdir

def read_data(name):
    f = "data/processed.%s.data" % (name)
    print("read %s" % f)
    col_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', \
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'risk']
    df = pd.read_csv(f, names = col_names)
    # check for ? missing values - six of 'em
    # check for just ? or any non-numeric character
    for c in col_names:
        df[c] = df[c].apply(lambda s: np.nan if s=='?' else s)
    df = df.dropna()
    df['Y'] = df['risk'].apply(lambda x: 1 if x>0 else 0)
    print("raw df shape %s head\n%s" % (df.shape, df[:3]))
#    print("raw df stats\n%s" % df.describe())
    return df

def load_data(name, print_out=True):
    df = read_data(name)
#    train = df[:210]
#    test  = df[210:]
    train, test = train_test_split(df, test_size=0.3, random_state=10)
    yvars = ['risk', 'Y']
    train_y = train[yvars]
    test_y  = test[yvars]
#    train_r = train['risk']    # use risk for five-way multi-class classification
    train = train.drop(['risk', 'Y'], axis=1)
    test  = test.drop(['risk', 'Y'],  axis=1)
    if print_out:
        print("train test shapes %s %s %s %s" % (train.shape, test.shape, train_y.shape, test_y.shape))
        print("train head\n%s" % (train[:3]))
        print("test head\n%s" % (test[:3]))
        print("train_y set %s, test_y set %s" % (set(train_y['Y']), set(test_y['Y'])))
        print("train_y stats\n%s\ntest_y stats\n%s" % (train_y.describe(), test_y.describe()))
    return train, test, train_y, test_y

def test_incoming(test_X, train_X):
    '''Pretend we have streaming data coming in.  Run a few
       statistical tests on it, see if it's close to training data.'''
    assert(0==0)   # to do

def plot_scatter_matrix(df, plotdir):
    plt.clf()
    pd_scatter_matrix(df)
    plt.savefig(plotdir + 'scatter_matrix.png')

def plot_hists(df, plotdir, ncols=3):
    plt.clf()
    vlist = df.columns
    print("vlist", vlist)
    nrows = len(vlist) // ncols
    if len(vlist) % ncols > 0:
        nrows += 1
    for i, var in enumerate(vlist):
        plt.subplot(nrows, ncols, i+1)
        plt.hist(df[var])   # bins=30
        plt.title(var, fontsize=10)
        plt.tick_params(labelbottom='off', labelleft='off')
    plt.savefig(plotdir + 'hist_coronary.png')

def confusion_report(dftest_y, new_y):
    print("classification report\n%s" % classification_report(dftest_y['Y'], new_y))
    cm = confusion_matrix(dftest_y['Y'], new_y)
    print("confusion matrix\n%s" % cm)

def fit_predict(clf, dftrain, dftrain_y, dftest, dftest_y, label='x'):
#    print("fit shapes", dftrain.shape, dftrain_y.shape, dftest.shape, dftest_y.shape)
    clf.fit(dftrain, dftrain_y['Y'])
    fit_score = clf.score(dftrain, dftrain_y['Y'])
    pred_score = clf.score(dftest, dftest_y['Y'])
    new_y = clf.predict(dftest)
    print("%s: fit score %.5f, predict score %.5f" % (label, fit_score, pred_score))
#    confusion_report(dftest_y, new_y)
    return pred_score

def main():
    train_X, test_X, train_y, test_y = load_data('cleveland', print_out=False)
#    plotdir = make_plotdir()
#    plot_scatter_matrix(train_X, plotdir)  # takes a while, not that useful 
#    plot_hists(train_X, plotdir)    # fails
    clf = lr()
    fit_predict(clf, train_X, train_y, test_X, test_y, label='logistic')
    clf = LinearSVC()   # why is score somewhat randomized?
    fit_predict(clf, train_X, train_y, test_X, test_y, label='svc')

if __name__ == '__main__':
    main()
