import unittest

import numpy as np

from dlnpy.layers.activations import Sigmoid, Softmax, ReLU


class TestSigmoid(unittest.TestCase):

    def setUp(self):
        self.activation = Sigmoid()
        self.x = 0
        self.loss = 1

    def test_forward(self):
        right = 0.5
        y = self.activation.forward(self.x)
        self.assertEqual(y, right)

    def test_backward(self):
        right = 0.25
        _ = self.activation.forward(self.x)
        g = self.activation.backward(self.loss)
        self.assertEqual(g, right)


class TestSoftmax(unittest.TestCase):

    def setUp(self):
        self.activation = Softmax()
        self.x = np.array([[0, 0], [1, 1], [2, 2]])
        self.loss = np.array([[.5, .5], [.5, .5], [.5, .5]])

    def test_forward(self):
        right = np.array([[.5, .5], [.5, .5], [.5, .5]])
        y = self.activation.forward(self.x)
        self.assertTrue(np.array_equal(y, right))

    def test_backward(self):
        right = np.array([[0, 0], [0, 0], [0, 0]])
        _ = self.activation.forward(self.x)
        g = self.activation.backward(self.loss)
        self.assertTrue(np.array_equal(g, right))


class TestReLU(unittest.TestCase):

    def setUp(self):
        self.activation = ReLU()
        self.x = np.array([-2, 0, 2])
        self.loss = np.array([1, 1, 1])

    def test_forward(self):
        right = np.array([0, 0, 2])
        y = self.activation.forward(self.x)
        self.assertTrue(np.array_equal(y, right))

    def test_backward(self):
        right = np.array([0, 0, 1])
        _ = self.activation.forward(self.x)
        g = self.activation.backward(self.loss)
        self.assertTrue(np.array_equal(g, right))


if __name__ == '__main__':
    unittest.main()
