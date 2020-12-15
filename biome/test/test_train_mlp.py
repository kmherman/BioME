"""
Unittest module to test train_mlp module.


"""
import unittest

import numpy as np
import torch
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
        output = biome.train_nn1(x, y_onehot, h_0=20)
        self.assertTrue(isinstance(output, tuple))

    def test_smoke2(self):
        """
        Smoke test on train_nn3 to ensure that tuple of parameters
        is returned from training function.
        """
        output = biome.train_nn3(x, y_onehot, h_0=10, h_1=10, h_2=10)
        self.assertTrue(isinstance(output, tuple))


if __name__ == '__main__':
    unittest.main()
