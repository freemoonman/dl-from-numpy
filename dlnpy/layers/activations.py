import numpy as np

from .base import Layer


class Sigmoid(Layer):

    @staticmethod
    def _function(x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self._y = self._function(x)
        return self._y

    def backward(self, gy, optimizer=None):
        return gy * (self._y * (1.0 - self._y))


class Softmax(Layer):

    @staticmethod
    def _function(x):
        exp_x = np.exp(x - x.max(axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, -1, keepdims=True)

    def forward(self, x):
        self._y = self._function(x)
        return self._y

    def backward(self, gy, optimizer=None):
        return self._y * (gy - np.sum(self._y * gy, -1, keepdims=True))


class TanH(Layer):

    @staticmethod
    def _function(x):
        return 2 / (1 + np.exp(-2*x)) - 1

    def forward(self, x):
        self._y = self._function(x)
        return self._y

    def backward(self, gy, optimizer=None):
        return gy * (1 - np.power(self._y, 2))


class ReLU(Layer):

    @staticmethod
    def _function(x):
        return np.where(x > 0, x, 0)

    def forward(self, x):
        self._y = self._function(x)
        return self._y

    def backward(self, gy, optimizer=None):
        return gy * np.where(self._y > 0, 1, 0)
