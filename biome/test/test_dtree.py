"""
This module contains unit tests for the Decision Tree
Classification (dtree) model.

test_smoketest(self)-- smoke test for dtree
test_edge1(self)-- does it catch the exception for x_train not an array
test_edge2(self)--does it catch the exception for y_train not an array
test_deg3)self)-- are the arrays of same shape?


This is just the unittest script to ensure
that the module and functions is correctly
"""

import unittest
import numpy as np
from sklearn.datasets import load_iris  # Sample dataset
import biome


class TestDtree(unittest.TestCase):
    """
    unit test for knn_regression module, this class is inhered child class of
    unittest. unittest.TestCase is parent class.
    """
    xdat, ydat = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = biome.split_train_test(xdat, ydat)

    def test_smoke(self):
        """
        test to see if it runs
        """
        model = biome.decision_tree(TestDtree.x_train,
                                    TestDtree.y_train)
        self.assertTrue(model)
        # do we get the two right answer wee were expecting?

    def test_edge1(self):
        """
        does it catch the exceptions that x_train is not np array?
        """
        edge1_x_train = TestDtree.x_train
        with self.assertRaises(TypeError):
            biome.decision_tree(edge1_x_train, TestDtree.y_train)

    def test_edge2(self):
        """
        does it catch the exceptions that y_train is not np array?
        """
        edge2_y_train = TestDtree.y_train
        with self.assertRaises(TypeError):
            biome.decision_tree(TestDtree.x_train, edge2_y_train)

    def test_edg3(self):
        """
        does it catch the value that y_train is not np array?
        """
        edge3_y_train = np.array([0, 0])
        with self.assertRaises(ValueError):
            biome.decision_tree(TestDtree.x_train, edge3_y_train)


if __name__ == '__main__':
    unittest.main()

SUITE = unittest.TestLoader().loadTestsFromTestCase(TestDtree)
_ = unittest.TextTestRunner().run(SUITE)
