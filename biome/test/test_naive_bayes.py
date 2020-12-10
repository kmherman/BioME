# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:14:46 2020

@author: user
"""

import unittest
from sklearn.datasets import load_iris
from biome import prep_split_data as psd
import naive_bayes as nb


class UnitTests(unittest.TestCase):
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = psd.split_train_test(x, y)

    y_pred = [2, 1, 0, 0, 2, 1, 1, 2, 0, 2, 1, 1, 1, 1, 1]
    model = nb.naive_bayes(x_train, y_train)
