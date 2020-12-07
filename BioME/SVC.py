# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:29:42 2020

@author: LT
"""
import numpy as np
from sklearn.svm import SVC
import sklearn.model_selection

## This will need data_ loader and split_train_test to work




"""
This module contains functions to train a supporter vector classifier (SVC)
model.

The get_SVC method takes training data and returns a trained and fitted SVC
model.
"""

import pandas as pd


OTUpath = 'C:\\Users\\user\\Desktop\\CSE583\\Project\\Red-Team-sckit-learn-GMB\\bug_OTU_rel.tsv'
metaPath = 'C:\\Users\\user\\Desktop\\CSE583\\Project\\Red-Team-sckit-learn-GMB\\FecesMeta.txt'

def data_loader(path_otu_table, path_metadata, column_number=2):
    """
    Load OTU table and metadata categorical data and convert to numpy array.
    ex. x_data, y_data = data_loader('../bug_OTU_rel.tsv', '../FecesMeta.txt')

    Parameters:
    path_otu_table = string containing path to OTU data table
    path_metadata = string containing path to metadata table with categorical
                    assignments
    column_number = integer specifying column containing categorical data
                    (default=2)

    Returns:
    x_data_np = numpy array containing OTU data (samples x OTU features)
    y_var_np = numpy array containing categorical labels for each sample
    """
    x_data_pd = pd.read_table(path_otu_table, index_col=0, skiprows=1).T
    y_var_pd = pd.read_table(path_metadata, index_col=0,
                             usecols=[0, column_number])
    x_data_np = x_data_pd.to_numpy()
    y_var_np = y_var_pd.to_numpy()
    return x_data_np, y_var_np


def get_one_hot(list_category, y_data, column_number=0):
    """
    Produce one-hot encoded vectors from categorical data
    ex. y_output = get_one_hot(['CD', 'UC', 'IC', 'HC', 'CC'], y_data)

    Parameters:
    list_category = list or 1-D array of strings containing labels for each
                    category
    y_data = numpy array containing categorical assignments for each sample
    column_number = integer specifying which column the categorical data is in
                    (default=0)

    Returns:
    one_hot_output = numpy array containing one-hot encoded classification data
                    (size #samples x #categories)
    """
    number_samples = np.size(y_data, axis=0)
    number_categories = len(list_category)
    one_hot_output = np.zeros((number_samples, number_categories))
    for i in range(number_samples):
        if y_data[i, column_number] in list_category:
            pass
        else:
            raise ValueError('Categorical label is not contained in list')
        for j in range(number_categories):
            if y_data[i, column_number] == list_category[j]:
                one_hot_output[i, j] += 1
            else:
                pass
    return one_hot_output


def split_train_test(x_data, y_data):
    """
    Function splits data and labels into test and train data (10/90 split)
    ex. x_train, x_test, y_train, y_test = split_train_test(x_data, y_data)

    Parameters:
    x_data = OTU table in form of numpy array with 16S sequences as the columns
    y_data = one-hot encoded categorical labels or unencoded y data

    Returns:
    x_train = Random 90% of OTU data
    x_test = Random 10% of OTU data
    y_train = Random 90% of one-hot encoded labels/y array
    y_test = Random 10% of one-hot encoded labels/y array
    """
    if np.size(x_data, axis=0) == np.size(y_data, axis=0):
        pass
    else:
        raise IndexError('x_data and y_data have different number of samples')
    number_samples = np.size(x_data, axis=0)
    rand_indices = np.random.permutation(number_samples)
    train_size = round(number_samples*0.9)
    train_indices = rand_indices[0:train_size]
    test_indices = rand_indices[train_size:]
    x_train = x_data[train_indices]
    x_test = x_data[test_indices]
    y_train = y_data[train_indices]
    y_test = y_data[test_indices]
    return 



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
    # Reshape y_data into vector:
    y_train = y_train.ravel()
    # Check data type:
    if isinstance(x_train, np.ndarray) == False or \
        isinstance(y_train, np.ndarray) == False:
        raise TypeError("Training data must be numpy arrays.")
    # Check shape:
    x_shape = np.shape(x_train)
    y_shape = np.shape(y_train)
    if x_shape[0] != y_shape[0]:
        raise ValueError("The number of rows in the X training data must be \
                         equal to the length of the Y training data vector. \
                         \n Rows in X: " + str(x_shape[0]) + \
                         "\n Length of Y: " + str(y_shape[0]))
    # Search and select best model parameters:
    model = SVC(kernel='rbf', probability = True)
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, \
                                                    0.00001, 10]}
    clf_grid = sklearn.model_selection.GridSearchCV(model, \
                                                    param_grid, verbose=1)
    # Train model:
    clf_grid.fit(x_train, y_train)
    model = SVC(C=clf_grid.best_params_['C'], kernel='rbf', \
                gamma = clf_grid.best_params_['gamma'])
    return model



x_data, y_data = data_loader(OTUpath, metaPath, column_number=2)
x_train, x_test, y_train, y_test= split_train_test(x_data, y_data)