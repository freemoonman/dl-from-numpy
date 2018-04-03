import numpy as np

from .base import Layer
from .utils import to_tuple, get_conv_shape, image2column, column2image


class _Conv(Layer):

    def __init__(self, rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='same',
                 use_bias=True,
                 **kwargs):
        super().__init__(**kwargs)

        self._rank = rank
        self._filters = filters
        self._kernel_size = to_tuple(kernel_size, rank, 'kernel_size')
        self._strides = to_tuple(strides, rank, 'strides')
        self._padding = padding
        self._use_bias = use_bias

        self._kernel = None
        self._bias = None
        self._dkernel = None
        self._dbias = None

        self._cx = None
        self._ckernel = None

    def forward(self, x):
        raise NotImplementedError

    def backward(self, gy, optimizer):
        raise NotImplementedError


class Conv2D(_Conv):

    def __init__(self, filters, kernel_size,
                 strides=1,
                 padding='same',
                 use_bias=True,
                 **kwargs):
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            **kwargs
        )

    def initialize(self):
        input_units = self._input_shape[-1]
        output_units = self._filters
        kernel_shape = self._kernel_size + (input_units, output_units)

        limit = np.sqrt(6 / (input_units + output_units))
        self._kernel = np.random.uniform(-limit, limit, size=kernel_shape)
        self._dkernel = np.zeros_like(self._kernel)

        if self._use_bias:
            self._bias = np.zeros((output_units, ))
            self._dbias = np.zeros_like(self._bias)

    def calc_output_shape(self):
        output_h, output_w, _, _ = get_conv_shape(
            self._input_shape,
            filter_shape=self._kernel_size,
            stride_shape=self._strides,
            padding=self._padding,
            rank=self._rank
        )
        self._output_shape = (self._input_shape[0],
                              output_h,
                              output_w,
                              self._filters)

    def calc_param_size(self):
        if self._use_bias:
            self._param_size = self._kernel.size + self._bias.size
        else:
            self._param_size = self._kernel.size

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        self._x = x
        self._cx = image2column(x,
                                filter_shape=self._kernel_size,
                                stride_shape=self._strides,
                                padding=self._padding,
                                rank=self._rank)
        self._ckernel = self._kernel.transpose((3, 2, 0, 1))
        self._ckernel = self._ckernel.reshape((self._filters, -1))
        y = np.dot(self._cx, self._ckernel.T)
        # Reshape into (batch_size, output_h, output_w, filters)
        y = y.reshape((batch_size, *self._output_shape[1:-1], self._filters))
        return y

    def backward(self, gy, optimizer):
        kernel_h, kernel_w, channels, filters = self._kernel.shape
        # Reshape gradient into column shape
        gy = gy.reshape(-1, self._filters)

        gkernel = np.dot(self._cx.T, gy)
        gkernel = gkernel.transpose((1, 0)).reshape((filters,
                                                     channels,
                                                     kernel_h,
                                                     kernel_w))
        # Reshape into (kernel_h, kernel_w, channels, filters)
        gkernel = gkernel.transpose((2, 3, 1, 0))
        gbias = np.sum(gy, axis=0)
        self._kernel, self._dkernel = \
            optimizer.update(self._kernel, self._dkernel, gkernel)
        self._bias, self._dbias = \
            optimizer.update(self._bias, self._dbias, gbias)

        cgy = np.dot(gy, self._ckernel)
        # Reshape from column shape to image shape
        gx = column2image(cgy,
                          tensor_shape=self._x.shape,
                          filter_shape=self._kernel_size,
                          stride_shape=self._strides,
                          padding=self._padding,
                          rank=self._rank)
        return gx
