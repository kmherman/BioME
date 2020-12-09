# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:14:46 2020

@author: user
"""

import unittest
from sklearn.datasets import load_iris
import biome.test_prep_split_data as tsp
#TODO fix this import
#from biome. import naive_bayes as nb

# Have they used any code that uses "import biome" since biome.py was added
# to the directory? Becauase when importing, it just runs the module and 
# prints out the ASCII label due to name ambiguity


#x, y = load_iris(return_X_y=True)
#x_train, x_test, y_train, y_test = biome.split_train_test(x, y)





# class UnitTests(unittest.TestCase):
    
  

#     y_pred = [2, 1, 0, 0, 2, 1, 1, 2, 0, 2, 1, 1, 1, 1, 1]
#     model = biome.naive_bayes.get_GNB(x_train, y_train, x_test, y_test)
    
#     def test_smoke(self):
#         """
#         This method tests that the constructor makes the correct data type.
#         """
#         model = nb.GNB()
#         self.assertTrue(isinstance(model, __main__.GNB))
    
#     def test_data_type(self):
#         """
#         This method tests that an error is thrown if data is incorrect type.
#         """
#         x_train = "A string"
#         model = nb.GNB()
#         with self.assertRaises(TypeError):
#             model.get_GNB(x_train, y_train, x_test, y_test)
    
#     def test_data_size(self):
#         """
#         This method tests that an error is thrown if data is incorrect size.
#         """
#         y_train = y_train[2:] # Make y_train too short
#         model = nb.GNB()
#         with self.assertRaises(ValueError):
#             model.get_GNB(x_train, y_train, x_test, y_test)
    
#     def test_predict(self):
#         """
#         This method tests that a trained model is mostly able to correctly
#         classify data.
#         """
#         model = nb.GNB()
#         model.get_GNB(x_train, y_train, x_test, y_test)
        
#         y_pred = model.predict(x_test)
        
#         n_incorrect = (y_test != y_pred).sum()
#         self.assertTrue(n_incorrect < 3)

# if __name__ == '__main__':
#     unittest.main()

# SUITE = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
# _ = unittest.TextTestRunner().run(SUITE)
