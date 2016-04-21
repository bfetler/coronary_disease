# -*- coding: utf-8 -*-

# coronary artery (heart) disease prediction

import os
import pandas as pd
from pandas.tools.plotting import scatter_matrix as pd_scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sst
from sklearn.preprocessing import StandardScaler
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
    "Read and clean data from file."
    f = "data/processed.%s.data" % (name)
    print("read %s" % f)
    col_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', \
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'risk']
    df = pd.read_csv(f, names = col_names)
    # check for ? missing values - six of 'em
    # check for just ? or any non-numeric character
    for c in col_names:
        df[c] = df[c].apply(lambda s: np.nan if s=='?' else float(s))
    df = df.dropna()
    # maybe columns w/ ? are str not int, need to convert
    df['Y'] = df['risk'].apply(lambda x: 1 if x>0 else 0)
    print("raw df shape %s head\n%s" % (df.shape, df[:3]))
#    print("raw df stats\n%s" % df.describe())
    return df

def load_data(name, print_out=True):
    "Read data and split into train, test data."
    df = read_data(name)
    train, test = train_test_split(df, test_size=0.3)  # random_state=10
    yvars = ['risk', 'Y']
    train_y = train[yvars]
    test_y  = test[yvars]
#    train_r = train['risk']    # for five-way multi-class classification
    train = train.drop(['risk', 'Y'], axis=1)
    test  = test.drop(['risk', 'Y'],  axis=1)
    if print_out:
        print("train test types %s %s %s %s" % (type(train), type(test), type(train_y), type(test_y)))
        print("train test shapes %s %s %s %s" % (train.shape, test.shape, train_y.shape, test_y.shape))
        print("train head\n%s" % (train[:3]))
        print("test head\n%s" % (test[:3]))
        print("train_y set %s, test_y set %s" % (set(train_y['Y']), set(test_y['Y'])))
        print("train_y stats\n%s\ntest_y stats\n%s" % (train_y.describe(), test_y.describe()))
    return train, test, train_y, test_y

def scale_data(train_X, test_X, print_out=False):
    "Scale data for transform, read dataframes, return nparrays."
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)  # nparray
    test_X  = scaler.transform(test_X)
    if print_out:
        print("scaler mean %s\nscaler std %s" % (scaler.mean_, scaler.scale_))
        print("train means %s" % [train_X[:, j].mean() for j in range(train_X.shape[1]) ])
        print("train std %s" % [train_X[:, j].std() for j in range(train_X.shape[1]) ])
        print("test means %s" % [test_X[:, j].mean() for j in range(test_X.shape[1]) ])
        print("test std %s" % [test_X[:, j].std() for j in range(test_X.shape[1]) ])
        # train_X columns are scaled by definition, test_X off by 1-2% (not surprising)
        print("train_X mean %.5f std %.5f" % (train_X.mean(), train_X.std()))
        print("test_X mean %.5f std %.5f" % (test_X.mean(), test_X.std()))
    # scaler.inverse_transform(train_X)
    return train_X, test_X

def test_incoming(test_X, train_X):
    '''Pretend we have streaming data coming in.  Run a few
       statistical tests on it, see if it's close to training data.'''
    vlist = list(train_X.columns)
    print("t-test compare train, test data\n    variable  pvalue   Pass")
    print("    --------  ------   ----")
    for var in vlist:
        pval = sst.ttest_ind(test_X[var], train_X[var]).pvalue
        print("    %-8s  %.4f   %s" % (var, pval, pval>0.05))
#       assert(pval > 0.05)  # in production, assert each column passes
# if only one or few values at a time, test where they are in train distribution
# is this a good idea?

def plot_scatter_matrix(df, plotdir):
    "Plot scatter matrix."
    plt.clf()
    pd_scatter_matrix(df)
    plt.savefig(plotdir + 'scatter_matrix.png')

def plot_hists(df, plotdir, label='x', ncols=3):
    "Plot histograms of data columns in one plot."
    plt.clf()
    vlist = list(df.columns)
    nrows = len(vlist) // ncols
    if len(vlist) % ncols > 0:
        nrows += 1
    for i, var in enumerate(vlist):
        plt.subplot(nrows, ncols, i+1)
        plt.hist(df[var].values, bins=15)
        plt.title(var, fontsize=9)
        plt.tick_params(labelbottom='off', labelleft='off')
    plt.savefig(plotdir + 'hist_coronary_' + label + '.png')

def confusion_report(test_y, new_y):
    "Create classification report and confusion matrix."
    print("classification report\n%s" % classification_report(test_y['Y'], new_y))
    cm = confusion_matrix(test_y['Y'], new_y)
    print("confusion matrix\n%s" % cm)

def fit_predict(clf, train_X, train_y, test_X, test_y, label='x'):
    "Fit train data, predict test data scores."
    clf.fit(train_X, train_y['Y'])
    fit_score = clf.score(train_X, train_y['Y'])
    pred_score = clf.score(test_X, test_y['Y'])
    print("%s: fit score %.5f, predict score %.5f" % (label, fit_score, pred_score))
#    new_y = clf.predict(test_X)
#    confusion_report(test_y, new_y)
    return pred_score

def cross_validate(clf, train_X, train_y, print_out=False):
    "Cross-validate fit scores.  Dataset is too small to be very reliable though."
    scores = cross_val_score(clf, train_X, train_y, cv=8)
    score = scores.mean()
    score_std = scores.std()
    if print_out:
        print("  CV scores mean %.4f +- %.4f" % (score, 2.0 * score_std))
        print("  CV raw scores", scores)
    return score

def main():
    train_X, test_X, train_y, test_y = load_data('cleveland', print_out=False)
    test_incoming(test_X, train_X)
    
    plotdir = make_plotdir()
#    plot_scatter_matrix(train_X, plotdir)  # takes a while, not that useful 
    plot_hists(train_X, plotdir, label='train')
    plot_hists(test_X, plotdir, label='test')
    
    train_X, test_X = scale_data(train_X, test_X)
    
    clf = lr()
    fit_predict(clf, train_X, train_y, test_X, test_y, label='logistic')
    cross_validate(clf, train_X, train_y['Y'], print_out=True)

    clf = LinearSVC()   # score somewhat randomized if not scaled
    fit_predict(clf, train_X, train_y, test_X, test_y, label='svc')
    cross_validate(clf, train_X, train_y['Y'], print_out=True)
    
    # repeated runs gives cv scores 0.7 to 0.9, 2*std 0.1 to 0.2
    # almost the same for lr, svc; depends on random train_test_split

if __name__ == '__main__':
    main()
