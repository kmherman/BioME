"""
This module contains unit tests for the naive_bayes.py module
"""


import unittest
from sklearn.datasets import load_iris  # A sample dataset for classification
from biome import prep_split_data as psd
from biome import naive_bayes as nb


class UnitTests(unittest.TestCase):

    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = psd(x, y)
    y_pred = [2, 1, 0, 0, 2, 1, 1, 2, 0, 2, 1, 1, 1, 1, 1]
    model = nb.get_GNB(x_train, y_train)

    def test_data_type(self):
        """
        This method tests that an error is thrown if data is incorrect type.
        """
        x_train = "A string"
        model = nb.GNB()
        with self.assertRaises(TypeError):
            model.get_GNB(x_train, UnitTests.y_train)

    def test_data_size(self):
        """
        This method tests that an error is thrown if data is incorrect size.
        """
        y_train = UnitTests.y_train[2:]  # Make y_train too short
        model = nb.GNB()
        with self.assertRaises(ValueError):
            model.get_GNB(UnitTests.x_train, y_train, UnitTests.x_test,
                          UnitTests.y_test)

    def test_predict(self):
        """
        This method tests that a trained model is mostly able to correctly
        classify data.
        """
        model = nb.GNB()
        model.get_GNB(UnitTests.x_train, UnitTests.y_train)
        y_pred = model.predict(UnitTests.x_test)
        n_incorrect = (UnitTests.y_test != y_pred).sum()
        self.assertTrue(n_incorrect < 3)


if __name__ == '__main__':
    unittest.main()

SUITE = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
_ = unittest.TextTestRunner().run(SUITE)
