import numpy as np

from .base import Layer


class Dense(Layer):

    def __init__(self, units,
                 use_bias=True,
                 **kwargs):
        super().__init__(**kwargs)

        self._units = units
        self._use_bias = use_bias

        self._kernel = None
        self._bias = None
        self._dkernel = None
        self._dbias = None

    def initialize(self):
        input_units = self._input_shape[-1]
        output_units = self._units
        kernel_shape = (input_units, output_units)

        limit = np.sqrt(6 / (input_units + output_units))
        self._kernel = np.random.uniform(-limit, limit, size=kernel_shape)
        self._dkernel = np.zeros_like(self._kernel)

        if self._use_bias:
            self._bias = np.zeros((output_units, ))
            self._dbias = np.zeros_like(self._bias)

    def calc_output_shape(self):
        self._output_shape = (*self._input_shape[:-1], self._units)

    def calc_param_size(self):
        if self._use_bias:
            self._param_size = self._kernel.size + self._bias.size
        else:
            self._param_size = self._kernel.size

    def forward(self, x):
        self._x = x
        return np.dot(self._x, self._kernel) + self._bias

    def backward(self, gy, optimizer):
        kernel = self._kernel
        gkernel = np.dot(self._x.T, gy)
        self._kernel, self._dkernel = \
            optimizer.update(self._kernel, self._dkernel, gkernel)
        if self._use_bias:
            gbias = np.sum(gy, axis=0)
            self._bias, self._dbias = \
                optimizer.update(self._bias, self._dbias, gbias)
        return np.dot(gy, kernel.T)
