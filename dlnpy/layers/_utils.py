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


def get_conv_output_shape(input_shape, filter_shape, stride_shape, rank=2):
    if rank == 2:
        batch_size, input_h, input_w, channels = input_shape
        filter_h, filter_w = filter_shape
        stride_h, stride_w = stride_shape
        padding_h, padding_w = _get_padding_shape(input_shape,
                                                  filter_shape,
                                                  stride_shape,
                                                  rank=2)

        assert (input_h + 2 * padding_h - filter_h) % stride_h == 0
        assert (input_w + 2 * padding_w - filter_w) % stride_w == 0

        output_h = int((input_h + 2 * padding_h - filter_h) / stride_h + 1)
        output_w = int((input_w + 2 * padding_w - filter_w) / stride_w + 1)

        return batch_size, output_h, output_w, channels

    else:
        raise NotImplementedError


def _get_tensor2matrix_indices(tensor_shape,
                               filter_shape,
                               stride_shape,
                               rank=2):
    if rank == 2:
        batch_size, input_h, input_w, channels = tensor_shape
        filter_h, filter_w = filter_shape
        stride_h, stride_w = stride_shape

        _, output_h, output_w, _ = get_conv_output_shape(tensor_shape,
                                                         filter_shape,
                                                         stride_shape,
                                                         rank=2)

        i0 = np.repeat(np.arange(filter_h), filter_w)
        i0 = np.tile(i0, channels)
        i1 = stride_h * np.repeat(np.arange(output_h), output_w)
        j0 = np.tile(np.arange(filter_w), filter_h * channels)
        j1 = stride_w * np.tile(np.arange(output_w), output_h)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(channels), filter_h * filter_w).reshape(-1, 1)

        return i, j, k
    else:
        raise NotImplementedError


def tensor_to_matrix(tensor, filter_shape, stride_shape, padding, rank=2):
    """
    transform tensor to matrix

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

        if padding == 'same':
            padding_h, padding_w = _get_padding_shape((input_h, input_w),
                                                      filter_shape,
                                                      stride_shape,
                                                      rank)
            pad_width = (
                (0, 0),  # batch_size
                (padding_h, padding_h),  # input_h
                (padding_w, padding_w),  # input_w
                (0, 0)  # channels
            )
            tensor = np.pad(tensor, pad_width, mode='constant')
        elif padding == 'valid':
            padding_h, padding_w = 0, 0
        else:
            raise ValueError

        i, j, k = _get_tensor2matrix_indices(tensor.shape,
                                             filter_shape,
                                             (padding_h, padding_w),
                                             stride_shape)

        mat = tensor[:, i, j, k]
        mat = mat.transpose(1, 2, 0).reshape(filter_h * filter_w * channels, -1)
        return mat
    else:
        raise NotImplementedError
