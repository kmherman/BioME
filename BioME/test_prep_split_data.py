import unittest

import numpy as np

from prep_split_data import get_one_hot


class TestPrepSplitData(unittest.TestCase):
    """
    Insert class documentation later
    """

    def test_shot1(self):
        """
        One-shot test on get_one_hot to check that one-hot encoding is
        correctly performed.
        """
        self.assertEqual(get_one_hot(['cat', 'dog'],
                                     np.array([['cat'], ['dog']])),
                         np.array([[1, 0], [0, 1]]))

    def test_edge1(self):
        """
        Edge test on get_one_hot to check that ValueError is raised
        when label in data is not present in list.
        """
        with self.assertRaises(ValueError):
            get_one_hot(['cat', 'dog'], np.array([['cat'], ['kitten']]))
