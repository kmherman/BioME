# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:03:01 2020

@author: user
"""

from sklearn.naive_bayes import GaussianNB
import prep_split_data as psd

from sklearn.datasets import load_iris

x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = psd.split_train_test(x, y)

gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train).predict(x_test)

print("Number of mislabeled points out of a total %d points : %d" \
      %(x_test.shape[0], (y_test != y_pred).sum()))
    
def gaussian_NB(x_train, y_train):
    """
    This method constructs and trains a Guassian Naive Bayes (GNB)
    classification model with user input training data.

    Parameters
    ----------
    x_train : training data as a numpy array
    y_train : training labels as a numpy array
    
    Returns
    -------
    A trained GNB model (sklearn.naive_bayes.GaussianNB)

    """
    gnb = GaussianNB()
    model = gnb.fit(x_train, y_train)
    return model