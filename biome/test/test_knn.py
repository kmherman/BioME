"""
This module contains unit tests for the K Nearest neighbors model.
class TestKNN(unittest.Testcase)  -tests for our KNN function from knn.py
"""


import unittest
from unittest.mock import patch
from sklearn.datasets import load_iris
import biome


class Test_KNN(unittest.TestCase):
    """test functions"""
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = biome.split_train_test(x, y)
    N_1 = 'ten'
    N_2 = 0
    N_3 = 2

    @patch('builtins.input', return_value=N_1)
    def test_invalid_input(self, mock_input):
        """
        Tests that a ValueError is raised when the input for number of
        neighbors is the wrong type (i.e. string instead of int)
        """
        x_train = Test_KNN.x_train
        y_train = Test_KNN.y_train
        self.assertRaises(ValueError, biome.knn, x_train, y_train)

    @patch('builtins.input', return_value=N_2)
    def test_edge_zero(self, mock_input):
        """
        Tests that a ValueError is raised when the integer input for
        number of neighbors is not positive
        """
        x_train = Test_KNN.x_train
        y_train = Test_KNN.y_train
        self.assertRaises(ValueError, biome.knn, x_train, y_train)

    @patch('builtins.input', return_value=N_3)
    def test_smoke(self, mock_input):
        """
        Smoke test for KNN. Checks to see if model is returned
        """
        result = biome.knn(Test_KNN.x_train, Test_KNN.y_train)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()

SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_KNN)
_ = unittest.TextTestRunner().run(SUITE)
