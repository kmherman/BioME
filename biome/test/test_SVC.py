import unittest
from sklearn.datasets import load_iris  # Sample dataset 
import biome

class UnitTests(unittest.TestCase):

    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = biome.split_train_test(x, y)

    def test_data_type(self):
        """
        This method tests that an error is thrown if data is incorrect type.
        """
        x_train = "A string"
        with self.assertRaises(TypeError):
            model = biome.get_SVC(x_train, UnitTests.y_train)

    def test_data_size(self):
        """
        This method tests that an error is thrown if data is incorrect size.
        """
        y_train = UnitTests.y_train[2:]  # Make y_train too short
        with self.assertRaises(ValueError):
            model = biome.get_SVC(UnitTests.x_train, y_train)


if __name__ == '__main__':
    unittest.main()

SUITE = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
_ = unittest.TextTestRunner().run(SUITE)
