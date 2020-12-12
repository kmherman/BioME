"""
This module contains functions to train a supporter vector classifier (SVC)
model.

The get_SVC method takes training data and returns a trained and fitted SVC
model.
"""


import numpy as np
from sklearn.svm import SVC
import sklearn.model_selection


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
    if isinstance(x_train, np.ndarray) is False or \
            isinstance(y_train, np.ndarray) is False:
        raise TypeError("Training data must be numpy arrays.")
    # Check shape:
    x_shape = np.shape(x_train)
    y_shape = np.shape(y_train)
    if x_shape[0] != y_shape[0]:
        raise ValueError("The number of rows in the X training data must be"
                         "equal to the length of the Y training vector."
                         "\n Rows in X: " + str(x_shape[0]) +
                         "\n Length of Y: " + str(y_shape[0]))
    # Search and select best model parameters:
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001,
                                                    0.00001, 10]}
    clf_grid = sklearn.model_selection.GridSearchCV(model,
                                                    param_grid, verbose=1)
    # Train model:
    clf_grid.fit(x_train, y_train)
    model = SVC(C=clf_grid.best_params_['C'], kernel='rbf',
                gamma=clf_grid.best_params_['gamma'])
    model.fit(x_train, y_train)
    return model
