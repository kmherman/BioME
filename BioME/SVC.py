# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:29:42 2020
@author: LT
"""
import numpy
from sklearn.svm import SVC
import sklearn.model_selection

# This will need data_ loader and split_train_test to work

"""
This module contains functions to train a supporter vector classifier (SVC)
model.
The get_SVC method takes training data and returns a trained and fitted SVC
model.
"""


def get_SVC(x_train, y_train):

    """
    This method trains and fits a Support Vector Classifier model.
    Parameters
    ----------
    x_train : Training data, numpy.ndarray of shape (I,J)
    y_train : Training labels, numpy.ndarry of shape (I,)
    Raises
    ------
    TypeError
        Raised if either or both of the input data is the incorrect type.
    ValueError
        Raised if the X and Y training data are incorrectly shaped.
    Returns
    -------
    model : A trained and fitted SVC model.
    """

    if isinstance(x_train, numpy.ndarray) is False or \
       isinstance(y_train, numpy.ndarray) is False:
        raise TypeError("Training data must be numpy arrays.")
    x_shape = numpy.shape(x_train)
    y_shape = numpy.shape(y_train)
    if x_shape[0] != y_shape[0]:
        raise ValueError("The number of rows in the X training data must be \
                         equal to the length of the Y training data vector. \
                         \n Rows in X: " + str(x_shape[0]) +
                         "\n Length of Y: " + str(y_shape[0]))
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': [1, 0.1, 0.01, 0.001,
                            0.00001, 10]}
    clf_grid = sklearn.model_selection.GridSearchCV(model,
                                                    param_grid, verbose=1)
    clf_grid.fit(x_train, y_train)
    model = SVC(C=clf_grid.best_params_['C'],
                kernel='rbf',
                gamma=clf_grid.best_params_['gamma'])
    return model
