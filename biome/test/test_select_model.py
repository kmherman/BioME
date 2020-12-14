"""
This module tests the functions in select_model.py.

TestSelectModel -- Class that contains unittest functions for the functions
in select_model.py.

test_smoke1 -- Smoke test on get_trained_models to check that a list is
returned and is of the correct length.

test_smoke2 -- Smoke test that checks that evaluate_rank_models runs
and returns a tuple containing the "best model".

test_smoke3 -- Smoke test that checks that a string is returned from
get_prediction and this label is contained in the category list.

test_edge1 -- Edge test on get_trained_models that checks that a ValueError
is raised when a model abbreviation inputted by the user is not a valid model.
"""
import unittest
import numpy as np
from sklearn.datasets import make_classification

import biome

x, y = make_classification()
x = np.array(x)
y = np.array(y)
y_data = []
for i in range(len(y)):
    if y[i] == 1:
        y_data.append(['cat'])
    else:
        y_data.append(['dog'])
y_np = np.array(y_data)
y_onehot = biome.get_one_hot(['cat', 'dog'], y_np)


class TestSelectModel(unittest.TestCase):
    """
    This class contains 3 smokes tests and an edge test for the functions:
    get_trained_models, evaluate_rank_models, and get_prediction. Since these
    algorithms each have their own unittest, the accuracy of these models will
    not be focused on.
    """
    def test_smoke1(self):
        """
        Smoke test on get_trained_models to check that a list is
        returned and is the correct length given the input.
        """
        model_list = ['mlp1', 'mlp3', 'dtree']
        models_out = biome.get_trained_models(x, y_onehot, model_list)
        self.assertTrue(isinstance(models_out, list))
        self.assertTrue(len(models_out) == len(model_list))

    def test_smoke2(self):
        """
        Smoke test on evaluate_rank_models to ensure that a tuple is returned.
        """
        x_train = x[0:80, :]
        x_test = x[80:, :]
        y_train = y_onehot[0:80, :]
        y_test = y_onehot[80:, :]
        model_list = ['mlp1', 'mlp3', 'dtree', 'rr']
        best_model_info = biome.evaluate_rank_models(x_train, y_train, x_test,
                                                     y_test, model_list)
        self.assertTrue(isinstance(best_model_info, tuple))

    def test_smoke3(self):
        """
        One-shot test on get_prediction to check that the prediction given
        is a string and is contained in the category list.
        """
        category_labels = ['cat', 'dog']
        x_train = x[1:80, :]
        x_test = x[80:, :]
        y_train = y_onehot[1:80, :]
        y_test = y_onehot[80:, :]
        query = x[0, :]
        model_list = ['mlp1']
        best_model_info = biome.evaluate_rank_models(x_train, y_train, x_test,
                                                     y_test, model_list)
        prediction = biome.get_prediction(query, best_model_info[0],
                                          best_model_info[1], category_labels)
        self.assertTrue(isinstance(prediction, str))
        self.assertTrue(prediction in category_labels)

    def test_edge1(self):
        """
        Edge test on get_trained_models to check that an exception is raised
        when a model abbreviation given is not an abbreviation for an
        available model.
        """
        with self.assertRaises(ValueError):
            biome.get_trained_models(x, y_onehot, ['mlp2'])
