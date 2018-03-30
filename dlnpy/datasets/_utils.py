import os
import sys
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError

import numpy as np


HOME_PATH = os.environ['HOMEPATH']
DATA_PATH = os.path.join(HOME_PATH, '.dlnpy', 'datasets')


def get_file(path):

    path = os.path.join(DATA_PATH, path)
    if os.path.exists(path):
        return path
    else:
        return None


def save_npz(path, **kwargs):

    path = os.path.join(DATA_PATH, path)

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    np.savez(path, **kwargs)


def download(url):

    def _progress(count, block_size, total_size):
        percent = round((count * block_size) / total_size * 100)
        sys.stdout.write(f'\rDownloading {filename} {percent} %')
        sys.stdout.flush()

    filename = url.split('/')[-1]
    path = os.path.join(DATA_PATH, filename)

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    try:
        try:
            urlretrieve(url, path, reporthook=_progress)
            print(f'\rDownloaded {filename}')
        except HTTPError:
            raise
        except URLError:
            raise
    except (Exception, KeyboardInterrupt):
        if os.path.exists(path):
            os.remove(path)
        raise

    return path
