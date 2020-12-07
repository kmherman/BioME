""" This module is to develop the base training model
    with decision tree from sklearn
    This module contain only one function
    decision_tree-- function that trains data
    with GridSearchCV and returns
    optimized parameters for model
    _______________________________
    WARNING: must use y_data.np created from
    data_loader function found in prep_split_data module
    using output from get_one_hot function  WILL NOT WORK
    _______________________________
    Author: SG
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def decision_tree(x_train, y_train):
    """
    decision tree model
    _____________
    Inputs:
    x_train: independent variables (OTUS)
        format: array(can be multidimensional array)
    y_train: dependent variable
        fromat: of array (1-D array only)
    ______________
    Return:
        model_out: trained DT model with the best parameters.
    """
    if isinstance(x_train, np.ndarray) is False:
        raise TypeError("Training data must be numpy arrays.")
    if isinstance(y_train, np.ndarray) is False:
        raise TypeError("Training data must be numpy arrays.")
    x_shape = np.shape(x_train)
    y_shape = np.shape(y_train)
    if x_shape[0] != y_shape[0]:
        raise ValueError("The number of rows in the X training data must be \
                         equal to the length of the Y training data vector. \
                         \n Rows in X: " + str(x_shape[0]) +
                         "\n Length of Y: " + str(y_shape[0]))
    param_grid = {'criterion': ['gini', 'entropy'],
                  'max_depth': np.arange(3, 25),
                  'max_features': ['auto', 'sqrt', 'log2']}
    # decision tree model
    dtree_model = DecisionTreeClassifier(random_state=0)
    # use gridsearch to test all values
    dtree_gscv = GridSearchCV(dtree_model, param_grid, cv=10)
    # fit model to data
    dtree_gscv.fit(x_train, y_train)
    # save best model from the fitted model parameters
    model = DecisionTreeClassifier(
                criterion=dtree_gscv.best_params_['criterion'],
                max_depth=dtree_gscv.best_params_['max_depth'],
                max_features=dtree_gscv.best_params_['max_features'],
                random_state=0)
    model_out = model.fit(x_train, y_train)
    return model_out
