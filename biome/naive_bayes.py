"""
This module contains a class for Gaussian Naive Bayes classification.
The user initializes a "blank" model with the command GNB().
Then the user can train the model with the method get_GNB() and predict data
with predict().

The quick_test() method returns the number of incorrectly predicted samples
and the total number of samples to do a preliminary check on the model.
"""


"""
This module contains a class for Gaussian Naive Bayes classification.
The user initializes a "blank" model with the command GNB().
Then the user can train the model with the method get_GNB() and predict data
with predict().

The quick_test() method returns the number of incorrectly predicted samples
and the total number of samples to do a preliminary check on the model.
"""
import numpy as np
from sklearn.naive_bayes import GaussianNB


class GNB():

    MODEL = GaussianNB()

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def get_GNB(self):
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
        GNB.y_train = GNB.y_train.ravel()
        # Check data type:
        if isinstance(GNB.x_train, np.ndarray) is False or \
                isinstance(GNB.y_train, np.ndarray) is False:
            raise TypeError("Training data must be numpy arrays.")
        # Check shape:
        x_shape = np.shape(GNB.x_train)
        y_shape = np.shape(GNB.y_train)
        if x_shape[0] != y_shape[0]:
            raise ValueError("The number of rows in the X training data must "
                             "be equal to the length of the Y training data "
                             "vector. \n"
                             "Rows in X: " + str(x_shape[0]) +
                             "\n Length of Y: " + str(y_shape[0]))
        trained_model = GaussianNB().fit(GNB.x_train, GNB.y_train)
        return trained_model

    def predict(x_test):
        y_pred = GNB.trained_model.predict(x_test)
        return y_pred

    def quick_test(self):
        print("Number of mislabeled points out of a total %d points : %d"
              "%(GNB.x_test.shape[0], (GNB.y_test != GNB.y_pred).sum())")
