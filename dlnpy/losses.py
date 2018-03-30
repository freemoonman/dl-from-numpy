import numpy as np

# @todo: 設定ファイル
epsilon = 1e-7


class CrossEntropy(object):

    def __call__(self, t, p):
        p = np.clip(p, epsilon, 1 - epsilon)
        h = -t * np.log(p) - (1 - t) * np.log(1 - p)
        return np.mean(np.mean(h, -1, keepdims=True))

    @staticmethod
    def gradient(t, p):
        p = np.clip(p, epsilon, 1 - epsilon)
        return -(t / p) + (1 - t) / (1 - p)


class SoftmaxCrossEntropy(object):

    def __call__(self, t, p):
        p = np.clip(p, epsilon, 1 - epsilon)
        h = -t * np.log(p) - (1 - t) * np.log(1 - p)
        return np.mean(np.mean(h, -1, keepdims=True))

    @staticmethod
    def forward(x):
        exp_x = np.exp(x - x.max(axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, -1, keepdims=True)

    @staticmethod
    def gradient(t, p):
        return p - t


categorical_crossentropy = CrossEntropy()
