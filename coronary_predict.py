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
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split, cross_val_score
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
#    orig_col_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'risk']
    col_names = ['age', 'sex', 'chest_pain', 'b_pressure', 'cholesterol', \
        'b_sugar_up', 'ecg_type', 'heart_rate', 'exer_angina', \
        'exer_depress', 'exer_slope', 'fluor_count', 'thal_defect', 'risk']
    df = pd.read_csv(f, names = col_names)
    for c in col_names:
        df[c] = df[c].apply(lambda s: np.nan if s=='?' else float(s))
        # '?' char is nan, columns with it are str, convert to numeric
    df = df.dropna()
    df['Y'] = df['risk'].apply(lambda x: 1 if x>0 else 0)
    print("raw df shape %s head\n%s" % (df.shape, df[:3]))
    stats = df.describe()
    print("raw df stats\n%s" % stats)
    print("raw df        std / mean\n%s" % ( stats.ix['std'] / stats.ix['mean']) )
    return df

def load_data(name, plotdir, print_out=True):
    "Read data and split into train, test data."
    df = read_data(name)
    train, test = train_test_split(df, test_size=0.3)
#   plot_scatter_matrix(train, plotdir)  # takes a while, not that useful 
    yvars = ['risk', 'Y']
    train_y = train[yvars]
    test_y  = test[yvars]
#   train_r = train['risk']    # for five-way multi-class classification
    train = train.drop(['risk', 'Y'], axis=1)
    test  = test.drop(['risk', 'Y'],  axis=1)
    if print_out:
        print("train test types %s %s %s %s" % (type(train), type(test), type(train_y), type(test_y)))
        print("train test shapes %s %s %s %s" % (train.shape, test.shape, train_y.shape, test_y.shape))
        print("train head\n%s" % (train[:3]))
        print("test head\n%s" % (test[:3]))
        print("train_y set %s, test_y set %s" % (set(train_y['Y']), set(test_y['Y'])))
        print("train_y stats\n%s\ntest_y stats\n%s" % (train_y.describe(), test_y.describe()))

#   drop_col = ['b_sugar_up']
#   print('dropping high std/mean columns', drop_col)
#   train = train.drop(drop_col, axis=1)
#   test  = test.drop(drop_col, axis=1)
#   drop_col = ['age','exer_slope']
#   print('dropping low importance columns', drop_col)
#   train = train.drop(drop_col, axis=1)
#   test  = test.drop(drop_col, axis=1)
    return train, test, train_y, test_y

def scale_data(train_X, test_X, cols=None, print_out=False):
    """Scale data for transform.  
       cols: optional parameter specifies columns to scale (default is all)."""
    scaler = StandardScaler()
    if cols:
        train_X[cols] = scaler.fit_transform(train_X[cols])  # dataframe
        test_X[cols]  = scaler.transform(test_X[cols])
    else:
        train_X = scaler.fit_transform(train_X)  # nparray
        test_X  = scaler.transform(test_X)
    if print_out:
        print("scaler mean %s\nscaler std %s" % (scaler.mean_, scaler.scale_))
        if cols:
            print("train_X mean \n%s train_X std \n%s" % (train_X.mean(), train_X.std()))
            print("test_X mean \n%s test_X std \n%s" % (test_X.mean(), test_X.std()))
        else:
            print("train_X mean %.5f std %.5f" % (train_X.mean(), train_X.std()))
            print("test_X mean %.5f std %.5f" % (test_X.mean(), test_X.std()))
        # train_X columns are scaled, test_X rescaled off by 1-2% (not too bad)
    return train_X, test_X

def one_hot_encode(train_X, test_X, cols):
    "create series of one-hot encoded variables from column list"
    for col in cols:
        vals = set(train_X[col].values)
        for val in vals:
            newcol = '%s%d' % (col, int(val))
            train_X[newcol] = train_X[col].apply(lambda x: 1 if x==val else 0)
            test_X[newcol] = test_X[col].apply(lambda x: 1 if x==val else 0)
    train_X = train_X.drop(cols, axis=1)
    test_X  = test_X.drop(cols, axis=1)
    return train_X, test_X

def test_incoming(test_X, train_X):
    '''Pretend we have streaming data coming in.  Run a few
       statistical tests on it, see if it's close to training data.'''
    vlist = list(train_X.columns)
    print("t-test compare train, test data\n    variable         pvalue   Pass")
    print("    ---------------  ------   ----")
    for var in vlist:
        pval = sst.ttest_ind(test_X[var], train_X[var]).pvalue
        print("    %-15s  %.3f    %s" % (var, pval, pval>0.05))
#       assert(pval > 0.05)  # in production, assert each column passes
#       also depends how many values in a distribution used to make t-test

def plot_scatter_matrix(df, plotdir):
    "Plot scatter matrix."
    print('plotting scatter matrix, this may take a while')
    plt.clf()
    pd_scatter_matrix(df, figsize=(16,16))
    plt.suptitle("Scatter Matrix", fontsize=14)
    plt.savefig(plotdir + 'scatter_matrix.png')

def plot_hists(df, plotdir, label='x', ncols=3):
    "Plot histograms of data columns in one plot."
    plt.clf()
    f = plt.figure(1)
    f.suptitle(label + " Data Histograms", fontsize=12)
    vlist = list(df.columns)
    nrows = len(vlist) // ncols
    if len(vlist) % ncols > 0:
        nrows += 1
    for i, var in enumerate(vlist):
        plt.subplot(nrows, ncols, i+1)
        plt.hist(df[var].values, bins=15)
        plt.title(var, fontsize=10)
        plt.tick_params(labelbottom='off', labelleft='off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(plotdir + 'hist_coronary_' + label.lower() + '.png')

def fit_predict(clf, train_X, train_y, test_X, test_y, label='x'):
    "Fit train data, predict test data scores."
    clf.fit(train_X, train_y['Y'])
    fit_score = clf.score(train_X, train_y['Y'])
    pred_score = clf.score(test_X, test_y['Y'])
    print("%s: fit score %.5f, predict score %.5f" % (label, fit_score, pred_score))
    return pred_score

def cross_validate(clf, train_X, train_y, cv=5, print_out=False):
    "Cross-validate fit scores.  Dataset is too small to be very reliable though."
    scores = cross_val_score(clf, train_X, train_y, cv=cv, scoring='f1')
# scoring = 'accuracy' | 'f1' | 'precision' | 'recall'
    score = scores.mean()
    score_std = scores.std()
    if print_out:
        print("  CV scores mean %.4f +- %.4f" % (score, score_std))
        print("  CV raw scores", scores)
    return score, scores

# logistic function: p(x) = 1 / (1 + exp(a1*x1 + a2*x2 + b))
def print_lr_coefs(clf, X_labels):
    "print logistic regression coefficients"    
    plist = ((lab, val) for lab, val in zip(X_labels, clf.coef_[0]))
    plist = sorted(plist, key=lambda e: np.abs(e[1]), reverse=True)
    plist = pd.Series(data = (e[1] for e in plist), index = (e[0] for e in plist))
    print("Columns by logistic fit importance (order depends on random split)\n%s" % plist)
    print("Intercept:", clf.intercept_[0])
# usually in top 4-5: ['fluor_count', 'thal_defect', 'sex', 'max_heart_rate']

def explore_pca(train_X):
    pca = PCA()
    pout = pca.fit(train_X)
    print("PCA explained variance ratio\n", pout.explained_variance_ratio_)
# no big advantage to removing components, it takes 8 PCA comps out of 13 to reach 80% total variance


def main():
    plotdir = make_plotdir()
    train_X, test_X, train_y, test_y = load_data('cleveland', plotdir, print_out=False)
#   X_labels = list(train_X.columns)
    test_incoming(test_X, train_X)
    
    plot_hists(train_X, plotdir, label='Train')
    plot_hists(test_X, plotdir, label='Test')
    
    scale_cols = ['age','b_pressure','cholesterol','heart_rate','exer_depress','fluor_count']
    train_X, test_X = scale_data(train_X, test_X, scale_cols)
#   one_hot_cols = ['chest_pain','ecg_type','exer_slope','thal_defect']
    one_hot_cols = ['chest_pain']
    train_X, test_X = one_hot_encode(train_X, test_X, one_hot_cols)
#   print('one hot encode train_X head\n', train_X[:3])
    X_labels = list(train_X.columns)
    
    clf = lr()
    fit_predict(clf, train_X, train_y, test_X, test_y, label='logistic')
    cross_validate(clf, train_X, train_y['Y'], print_out=True)
    print_lr_coefs(clf, X_labels)

    clf = LinearSVC()   # data must first be scaled
    fit_predict(clf, train_X, train_y, test_X, test_y, label='svc')
    cross_validate(clf, train_X, train_y['Y'], print_out=True)
    
    explore_pca(train_X)
    
# repeated runs gives cv scores 0.7 to 0.9, std 0.04 to 0.08
# almost the same for lr, svc; depends on random train_test_split

if __name__ == '__main__':
    main()
