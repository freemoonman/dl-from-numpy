class Layer(object):

    def __init__(self, input_shape=None):
        if input_shape is None:
            self._input_shape = None
        else:
            self._input_shape = (None, *input_shape)
        self._output_shape = None
        self._param_size = None

        self._x = None
        self._y = None

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

    def calc_output_shape(self):
        self._output_shape = self._input_shape

    def calc_param_size(self):
        self._param_size = 0

    @property
    def input_shape(self):
        return self._input_shape

    @input_shape.setter
    def input_shape(self, input_shape):
        self._input_shape = input_shape

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def param_size(self):
        return self._param_size

    def forward(self, x):
        raise NotImplementedError

    def backward(self, gy, optimizer):
        raise NotImplementedError
