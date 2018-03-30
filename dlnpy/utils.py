import numpy as np


def train_test_split(x, y, test_size=0.2, shuffle=True, seed=None):

    assert x.shape[0] == y.shape[0]

    if seed:
        np.random.seed(seed)

    if shuffle:
        indices = np.random.permutation(x.shape[0])
        x = x[indices]
        y = y[indices]

    if 0 < test_size < 1:
        index = int(x.shape[0] * test_size)
    elif test_size >= 1 and isinstance(test_size, int):
        index = test_size
    else:
        # @todo: msg
        raise ValueError

    return x[:-index], x[-index:], y[:-index], y[-index:]


def to_categorical(y, num_classes=None):

    labels = sorted(set(y))
    if not num_classes:
        num_classes = len(labels)

    assert num_classes == len(labels)
    assert labels == [i for i in range(len(labels))]

    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1

    return one_hot
