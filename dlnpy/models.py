import time

import numpy as np

from .metrics import get as get_metrics


class Sequential(object):

    def __init__(self, layers=None):
        self._layers = []
        self._optimizer = None
        self._loss = None
        self._metrics = None

        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        if self._layers:
            layer.input_shape = self._layers[-1].output_shape
        layer.calc_output_shape()
        if hasattr(layer, 'initialize'):
            layer.initialize()
        layer.calc_param_size()

        self._layers.append(layer)

    def summary(self):
        print('')
        print(f'InputShape: {str(self._layers[0].input_shape)}')
        print(
            f'LayerType       OutputShape             Param #         '
        )
        print(
            f'========================================================'
        )
        total_param_size = 0
        for layer in self._layers:
            print(
                f'{str(layer):<16}'
                f'{str(layer.output_shape):<24}'
                f'{int(layer.param_size):<16}'
            )
            total_param_size += layer.param_size
        print(
            f'========================================================'
        )
        print(f'TotalParams: {int(total_param_size)}')
        print('')

    def compile(self, optimizer, loss, metrics=None):
        self._optimizer = optimizer
        self._loss = loss
        if metrics is not None:
            self._metrics = get_metrics(metrics, self._loss)

    def _forward(self, x):
        for layer in self._layers:
            x = layer.forward(x)

        return x

    def _backward(self, grad):
        for layer in reversed(self._layers):
            grad = layer.backward(grad, self._optimizer)

    def train_on_batch(self, x, t):
        y = self._forward(x)
        if hasattr(self._loss, 'forward'):
            y = self._loss.forward(y)
        loss = self._loss(t, y)

        grad = self._loss.gradient(t, y)
        self._backward(grad)

        if self._metrics is not None:
            metrics = self._metrics(t, y)
            return loss, metrics
        else:
            return loss, None

    def test_on_batch(self, x, t):
        y = self._forward(x)
        if hasattr(self._loss, 'forward'):
            y = self._loss.forward(y)
        loss = self._loss(t, y)

        if self._metrics is not None:
            metrics = self._metrics(t, y)
            return loss, metrics
        else:
            return loss, None

    def _show_info(self, validation_data, start, history):
        if validation_data is not None and self._metrics is not None:
            print(
                f'time: {int(time.time() - start)} s, '
                f'loss: {float(np.mean(history["loss"])):.4f}, '
                f'metrics: {float(np.mean(history["metrics"])):.4f}, '
                f'val_loss: {float(np.mean(history["val_loss"])):.4f}, '
                f'val_metrics: {float(np.mean(history["val_metrics"])):.4f}'
            )
        elif validation_data is not None:
            print(
                f'time: {int(time.time() - start)} s, '
                f'loss: {float(np.mean(history["loss"])):.4f}, '
                f'val_loss: {float(np.mean(history["val_loss"])):.4f}'
            )
        elif self._metrics is not None:
            print(
                f'time: {int(time.time() - start)} s,'
                f'loss: {float(np.mean(history["loss"])):.4f}, '
                f'metrics: {float(np.mean(history["metrics"])):.4f}'
            )
        else:
            print(
                f'time: {int(time.time() - start)} s,'
                f'loss: {float(np.mean(history["loss"])):.4f}'
            )

    def fit(self, x, y, batch_size=32, epochs=10, validation_data=None):
        assert x.shape[0] == y.shape[0]

        train_size = x.shape[0]
        train_batch_steps = train_size // batch_size
        if train_size % batch_size != 0:
            train_has_tail = True
            train_batch_steps += 1
        else:
            train_has_tail = False

        if validation_data is not None:
            x_ = validation_data[0]
            y_ = validation_data[1]

            assert x_.shape[0] == y_.shape[0]

            test_size = x_.shape[0]
            test_batch_steps = test_size // batch_size
            if train_size % batch_size != 0:
                test_has_tail = True
                test_batch_steps += 1
            else:
                test_has_tail = False
        else:
            x_ = None
            y_ = None
            test_batch_steps = 0
            test_has_tail = False

        for epoch in range(epochs):

            print(f'Epoch: {epoch+1}/{epochs}')
            start = time.time()

            indices = np.random.permutation(train_size)
            x = x[indices]
            y = y[indices]

            epoch_history = {'loss': []}
            if validation_data is not None:
                epoch_history['val_loss'] = []
            if self._metrics is not None:
                epoch_history['metrics'] = []
                if validation_data is not None:
                    epoch_history['val_metrics'] = []

            for step in range(train_batch_steps):
                if train_has_tail and step == train_batch_steps - 1:
                    train_x = x[step * batch_size:]
                    train_y = y[step * batch_size:]
                else:
                    train_x = x[step * batch_size:(step + 1) * batch_size]
                    train_y = y[step * batch_size:(step + 1) * batch_size]
                loss, metrics = self.train_on_batch(train_x, train_y)
                epoch_history['loss'].append(loss)
                if metrics is not None:
                    epoch_history['metrics'].append(metrics)

            if validation_data is not None:
                for step in range(test_batch_steps):
                    if test_has_tail and step == test_batch_steps - 1:
                        test_x = x_[step * batch_size:]
                        test_y = y_[step * batch_size:]
                    else:
                        test_x = x_[step * batch_size:(step + 1) * batch_size]
                        test_y = y_[step * batch_size:(step + 1) * batch_size]
                    val_loss, val_metrics = self.test_on_batch(test_x, test_y)
                    epoch_history['val_loss'].append(val_loss)
                    if val_metrics is not None:
                        epoch_history['val_metrics'].append(val_metrics)

            self._show_info(validation_data, start, epoch_history)

        return self

    def predict(self, x):
        y = self._forward(x)
        if hasattr(self._loss, 'forward'):
            y = self._loss.forward(y)

        return y
