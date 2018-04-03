import numpy as np


def to_tuple(value, n, name):
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            msg = f'The `{name}` argument must be a tuple of {n} integers. ' \
                  f'Received: {str(value)}'
            raise ValueError(msg)
        if len(value_tuple) != n:
            msg = f'The `{name}` argument must be a tuple of {n} integers. ' \
                  f'Received: {str(value)}'
            raise ValueError(msg)
        for single_value in value_tuple:
            try:
                int(single_value)
            except ValueError:
                msg = f'The `{name}` argument must be ' \
                      f'a tuple of {n} integers. ' \
                      f'Received: {str(value)} including element ' \
                      f'{str(single_value)} of type {str(type(single_value))}'
                raise ValueError(msg)
    return value_tuple


def _get_padding_shape(input_shape, filter_shape, stride_shape, rank=2):
    if rank == 2:
        input_h, input_w = input_shape
        filter_h, filter_w = filter_shape
        stride_h, stride_w = stride_shape

        assert ((input_h - 1) * stride_h - input_h + filter_h) % 2 == 0
        assert ((input_w - 1) * stride_w - input_w + filter_w) % 2 == 0

        padding_h = int(((input_h - 1) * stride_h - input_h + filter_h) / 2)
        padding_w = int(((input_w - 1) * stride_w - input_w + filter_w) / 2)

        return padding_h, padding_w
    else:
        raise NotImplementedError


def get_conv_shape(input_shape,
                   filter_shape,
                   stride_shape,
                   padding,
                   rank=2):
    if rank == 2:
        batch_size, input_h, input_w, channels = input_shape
        filter_h, filter_w = filter_shape
        stride_h, stride_w = stride_shape
        if padding == 'same':
            padding_h, padding_w = _get_padding_shape((input_h, input_w),
                                                      filter_shape,
                                                      stride_shape,
                                                      rank=2)
        elif padding == 'valid':
            padding_h, padding_w = 0, 0
        else:
            raise ValueError

        assert (input_h + padding_h - filter_h) % stride_h == 0
        assert (input_w + padding_w - filter_w) % stride_w == 0

        output_h = int((input_h + 2 * padding_h - filter_h) / stride_h + 1)
        output_w = int((input_w + 2 * padding_w - filter_w) / stride_w + 1)

        return output_h, output_w, padding_h, padding_w

    else:
        raise NotImplementedError


def image2column(tensor, filter_shape, stride_shape, padding, rank=2):
    """
    transform tensor to column

    :param tensor: np.array, shape order must be (batch_size, ..., channels)
    :param filter_shape: tuple
    :param stride_shape: tuple
    :param padding: str
    :param rank: int
    :return:
    """

    assert tensor.ndim - 2 == rank
    assert len(filter_shape) == rank

    if rank == 2:
        batch_size, input_h, input_w, channels = tensor.shape
        filter_h, filter_w = filter_shape
        stride_h, stride_w = stride_shape

        output_h, output_w, padding_h, padding_w = get_conv_shape(tensor.shape,
                                                                  filter_shape,
                                                                  stride_shape,
                                                                  padding,
                                                                  rank=2)

        if padding_h or padding_w:
            pad_width = (
                (0, 0),  # batch_size
                (padding_h, padding_h),  # input_h
                (padding_w, padding_w),  # input_w
                (0, 0)  # channels
            )
            tensor = np.pad(tensor, pad_width, mode='constant')

        col = np.empty((output_h * output_w * batch_size,
                        filter_h * filter_w * channels))
        c = 0
        for b in range(batch_size):
            for h in range(output_h):
                i = h * stride_h
                for w in range(output_w):
                    j = w * stride_w
                    tmp = tensor[b, i:i+filter_h, j:j+filter_w, :]
                    tmp = tmp.transpose((2, 0, 1)).flatten()
                    col[c, :] = tmp
                    c += 1

        return col
    else:
        raise NotImplementedError


def column2image(col, tensor_shape, filter_shape, stride_shape, padding, rank=2):
    """
    transform column to tenor

    :param col: np.array, shape order must be (batch_size, ..., channels)
    :param tensor_shape: tuple
    :param filter_shape: tuple
    :param stride_shape: tuple
    :param padding: str
    :param rank: int
    :return:
    """

    assert len(tensor_shape) - 2 == rank

    if rank == 2:
        batch_size, input_h, input_w, channels = tensor_shape
        filter_h, filter_w = filter_shape
        stride_h, stride_w = stride_shape

        output_h, output_w, padding_h, padding_w = get_conv_shape(tensor_shape,
                                                                  filter_shape,
                                                                  stride_shape,
                                                                  padding,
                                                                  rank=2)

        tensor_h, tensor_w = input_h + padding_h * 2, input_w + padding_w * 2
        tensor = np.empty((batch_size, tensor_h, tensor_w, channels))

        c = 0
        for b in range(batch_size):
            for h in range(output_h):
                i = h * stride_h
                for w in range(output_w):
                    j = w * stride_w
                    tmp = col[c, :].reshape(channels, filter_h, filter_w)
                    tmp = tmp.transpose((1, 2, 0))
                    tensor[b, i:i+filter_h, j:j+filter_w, :] = tmp
                    c += 1

        return tensor[:,
                      padding_h:input_h+padding_h,
                      padding_w:input_w+padding_w,
                      :]
    else:
        raise NotImplementedError
