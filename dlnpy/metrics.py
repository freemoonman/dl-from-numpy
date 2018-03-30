import numpy as np

from .losses import CrossEntropy


def categorical_accuracy(t, p):
    p = np.argmax(p, axis=-1)
    t = np.argmax(t, axis=-1)
    return np.mean(p == t)


def get(identifier, loss):
    if identifier == 'accuracy' and isinstance(loss, CrossEntropy):
        return categorical_accuracy
    else:
        raise NotImplementedError
