# -*- coding: utf-8 -*-

# coronary artery (heart) disease prediction

import os
import pandas as pd
from pandas.tools.plotting import scatter_matrix as pd_scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split, cross_val_score

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
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'Y']
    df = pd.read_csv(f, names = col_names)
    print("raw df shape %s head\n%s" % (df.shape, df[:3]))
    print("raw df stats\n%s" % df.describe())
    # check for ? missing values
    return df

def load_data(name, print_out=True):
    df = read_data(name)
#    train = df[:210]
#    test  = df[210:]
    train, test = train_test_split(df, test_size=0.3, random_state=10)
    train_y = train['Y']
    test_y  = test['Y']
    train = train.drop(['Y'], axis=1)
    test  = test.drop(['Y'],  axis=1)
    if print_out:
        print("train test shapes %s %s %s %s" % (train.shape, test.shape, train_y.shape, test_y.shape))
        print("train head\n%s" % (train[:3]))
        print("test head\n%s" % (test[:3]))
    return train, test, train_y, test_y

def test_incoming(test_X, train_X):
    '''Pretend we have streaming data coming in.  Run a few
       statistical tests on it, see if it's close to training data.'''
    assert(0==0)   # to do

def plot_scatter_matrix(df, plotdir):
    plt.clf()
    pd_scatter_matrix(df)
    plt.savefig(plotdir + 'scatter_matrix.png')

def main():
    train_X, test_X, train_y, test_y = load_data('cleveland')
    plotdir = make_plotdir()
#    plot_scatter_matrix(train_X, plotdir)  # takes a while, not that useful 

if __name__ == '__main__':
    main()
