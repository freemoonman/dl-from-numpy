class SGD(object):

    def __init__(self, lr=0.01, momentum=0):
        self._lr = lr
        self._momentum = momentum

    def update(self, param, dparam, grad):
        dparam = - self._lr * grad + self._momentum * dparam
        return param + dparam, dparam
