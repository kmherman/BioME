"""
Unittest module to test train_mlp module.


"""
import os
import unittest

import numpy as np
import torch

import biome

#x_in = torch.tensor([[2, 0], [0, 2]])
#w0 = torch.tensor([[2, 1], [1, 2]])
#b0 = torch.tensor([[2, 1], [2, 1]])                
#w1 = torch.tensor([[1, 2], [1, 2]])
#b1 = torch.tensor([[-13, -14], [-13, -12]])
#print(biome.forward_nn1(x_in, w0, w1, b0, b1))

class TestTrainMLP(unittest.TestCase):
    """
    Class to test functions in module train_mlp.
    """

    def test_shot1(self):
        """
        One-shot test on forward_nn1 to check that returned tensor
        is correct.
        """
        x_in = torch.tensor([[2, 0], [0, 2]])
        w0 = torch.tensor([[2, 1], [1, 2]])
        b0 = torch.tensor([[2, 1], [2, 1]])
        w1 = torch.tensor([[1, 2], [1, 2]])
        b1 = torch.tensor([[-11, -12], [-14, -13]])
        self.assertTrue((biome.forward_nn1(x_in, w0, w1, b0, b1)
                         == torch.tensor([[1, 0], [0, 1]])).all())

    def test_shot2(self):
        """
        One-shot test on forward_nn3 to check that returned tensor
        is correct.
        """
        x_in = torch.tensor([[2, 0], [0, 2]])
        w0 = torch.tensor([[2, 1], [1, 2]])
        b0 = torch.tensor([[2, 1], [2, 1]])
        w1 = torch.tensor([[1, 2], [1, 2]])
        b1 = torch.tensor([[-10, -10], [-12, -11]])
        w2 = torch.tensor([[1, 1], [1, 1]])
        b2 = torch.tensor([[1, 0], [0, 1]])
        w3 = torch.tensor([[1, 2], [2, 1]])
        b3 = torch.tensor([[-12, -14], [-17, -15]])
        self.assertTrue((biome.forward_nn3(x_in, w0, w1, w2, w3, b0, b1, b2,
                                          b3)
                         == torch.tensor([[1, 0], [0, 1]])).all())

    def test_smoke1(self):
        """
        Smoke test on train_nn1 to ensure that tuple of parameters
        is returned from training function.
        """
        x_data = np.array([[0, 1, 5, 10], [7, 2, 4, 7]])
        y_data = np.array([[1, 0], [0, 1]])
        output = biome.train_nn1(x_data, y_data, h_0=2)
        self.assertTrue(isinstance(output, tuple))

    def test_smoke2(self):
        """
        Smoke test on train_nn3 to ensure that tuple of parameters
        is returned from training function.
        """
        x_data = np.array([[0, 1, 5, 10], [7, 2, 4, 7]])
        y_data = np.array([[1, 0], [0, 1]])
        output = biome.train_nn3(x_data, y_data, h_0=2, h_1=2, h_2=2)
        self.assertTrue(isinstance(output, tuple))


if __name__ == '__main__':
    unittest.main()
