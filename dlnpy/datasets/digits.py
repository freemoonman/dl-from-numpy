import os
import numpy as np

from .utils import get_file, download, save_npz


def load_data(path='digits.npz'):

    def read_origin_data(fname):
        data = np.loadtxt(fname, delimiter=',')
        data = data.astype(np.uint8)
        return data[:, :-1], data[:, -1]

    full_path = get_file(path)

    if full_path is None:
        train_path = download(
            'http://archive.ics.uci.edu/'
            'ml/machine-learning-databases/optdigits/optdigits.tra'
        )
        test_path = download(
            'http://archive.ics.uci.edu/'
            'ml/machine-learning-databases/optdigits/optdigits.tes'
        )
        x_train, y_train = read_origin_data(train_path)
        x_test, y_test = read_origin_data(test_path)
        save_npz(path,
                 x_train=x_train, y_train=y_train,
                 x_test=x_test, y_test=y_test)
        os.remove(train_path)
        os.remove(test_path)
        full_path = get_file(path)

    f = np.load(full_path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()

    return (x_train, y_train), (x_test, y_test)
