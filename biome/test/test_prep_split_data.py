"""
Unittest module to test prep_split_data module..


TestPrepSplitData(unittest.TestCase) -- class containing a smoke test,
two one-shot tests, and two edge tests for prep_split_data.

test_smoke(self) -- function tests that data_loader returns a tuple of arrays.

test_shot1(self) -- function tests that get_one_hot returns correct one-hot
encoding for a simple example.

test_edge1(self) -- function tests that get_one_hot raises an exception when a
categorical label in y_data is not contained in specified category list.

test_shot2(self) -- function tests that split_train_test returns split arrays
of appropriate sizes (90/10 train/test).

test_edge2(self) -- function tests that split_train_test raises an exception
when the x and y data have a different number of samples.
"""
import os
import unittest

import numpy as np

import BioME


data_path = os.path.join(BioME.__path__[0], 'Data')


class TestPrepSplitData(unittest.TestCase):
    """
    Class to test functions in module prep_split_data.
    """

    def test_smoke(self):
        """
        Smoke test on data_loader to ensure that a tuple (of two arrays)
        is returned.
        """
        path_OTU = os.path.join(data_path, 'bug_OTU_rel.tsv')
        path_meta = os.path.join(data_path, 'FecesMeta.txt')
        self.assertTrue(isinstance(BioME.data_loader(
                        path_OTU, path_meta), tuple))

    def test_shot1(self):
        """
        One-shot test on get_one_hot to check that one-hot encoding is
        correctly performed.
        """
        self.assertTrue((BioME.get_one_hot(['cat', 'dog'],
                                           np.array([['cat'],
                                                     ['dog']]))
                         == np.array([[1, 0], [0, 1]])).all())

    def test_edge1(self):
        """
        Edge test on get_one_hot to check that ValueError is raised when label
        in data is not present in list.
        """
        with self.assertRaises(ValueError):
            BioME.get_one_hot(['cat', 'dog'], np.array([['kitten']]))

    def test_shot2(self):
        """
        One-shot test on split_train_test to ensure 90/10 split of data.
        """
        xtrain, xtest, ytrain, ytest = BioME.split_train_test(
                                                        np.array([[0], [1],
                                                                  [2], [3],
                                                                  [4], [5],
                                                                  [6], [7],
                                                                  [8], [9]]),
                                                        np.array([[0], [1],
                                                                  [2], [3],
                                                                  [4], [5],
                                                                  [6], [7],
                                                                  [8], [9]]))
        self.assertTrue(np.size(ytrain, axis=0) == 9)
        self.assertTrue(np.size(xtrain, axis=0) == 9)
        self.assertTrue(np.size(ytest, axis=1) == 1)
        self.assertTrue(np.size(xtest, axis=1) == 1)

    def test_edge2(self):
        """
        Edge test on split_train_test to check that exception is raised when
        x_data and y_data do not have the same number of samples.
        """
        with self.assertRaises(IndexError):
            BioME.split_train_test(np.array([[0], [1], [2], [3],
                                                       [4], [5], [6], [7],
                                                       [8], [9]]),
                                   np.array([[0], [1], [2], [3],
                                             [4], [5], [6], [7], [8]]))


if __name__ == '__main__':
    unittest.main()
