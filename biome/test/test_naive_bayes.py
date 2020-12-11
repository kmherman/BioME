"""
This module contains unit tests for the naive_bayes.py module
"""


import unittest
from sklearn.datasets import load_iris  # A sample dataset for classification
import biome


class UnitTests(unittest.TestCase, biome.naive_bayes.GNB):

    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = biome.split_train_test(x, y)

    def test_data_type(self):
        """
        This method tests that an error is thrown if data is incorrect type.
        """
        x_train = "A string"
        model = biome.GNB(x_train, UnitTests.y_train)
        with self.assertRaises(TypeError):
            model.get_GNB(x_train, UnitTests.y_train)

    def test_data_size(self):
        """
        This method tests that an error is thrown if data is incorrect size.
        """
        y_train = UnitTests.y_train[2:]  # Make y_train too short
        model = biome.GNB(UnitTests.x_train, y_train)
        with self.assertRaises(ValueError):
            model.get_GNB()

    def test_predict(self):
        """
        This method tests that a trained model is mostly able to correctly
        classify data.
        """
        model = biome.GNB(UnitTests.x_train, UnitTests.y_train)
        model.get_GNB()
        y_pred = model.predict(UnitTests.x_test)
        n_incorrect = (UnitTests.y_test != y_pred).sum()
        self.assertTrue(n_incorrect < 3)


if __name__ == '__main__':
    unittest.main()

SUITE = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
_ = unittest.TextTestRunner().run(SUITE)
