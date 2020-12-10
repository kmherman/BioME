# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:03:01 2020

@author: user
"""
import numpy as np
from sklearn.naive_bayes import GaussianNB
from prep_split_data import split_train_test

from sklearn.datasets import load_iris

x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = split_train_test(x, y)


class GNB():

    MODEL = GaussianNB()

    def __init__(self, x_train, y_train, x_test, y_test, gnb=GaussianNB()):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_GNB(x_train, y_train):
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

        # Reshape y_data into vector:
        GNB.y_train = y_train.ravel()
        # Check data type:
        if isinstance(x_train,
                      np.ndarray) is False or isinstance(y_train,
                                                         np.ndarray) is False:
            raise TypeError("Training data must be numpy arrays.")
        # Check shape:
        x_shape = np.shape(x_train)
        y_shape = np.shape(y_train)
        if x_shape[0] != y_shape[0]:
            raise ValueError("The number of rows in the X training data must"
                             "be equal to the length of the Y training data"
                             " vector."
                             "\n Rows in X: " + str(x_shape[0]) +
                             "\n Length of Y: " + str(y_shape[0]))
        model = GNB.MODEL.fit(x_train, y_train)
        return model

    def predict_GNB():
        y_pred = GNB.MODEL.predict(GNB.x_test)
        return y_pred

    def test_model():
        print("Number of mislabeled points out of a total %d points : %d"
              % (x_test.shape[0], (GNB.y_test != GNB.y_pred).sum()))
