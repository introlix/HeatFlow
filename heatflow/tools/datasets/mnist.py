import os
import gzip
import requests
from typing import Tuple

import numpy as np
from tqdm import tqdm
from .get_datasets import getData
from heatflow.utils import to_categorical

def getMNIST() -> Tuple[np.ndarray]:

    URLS = [
        (
            "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
            "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
            60000,
        ),
        (
            "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
            "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
            10000,
        ),
    ]

    data = list()

    for img_url, label_url, size in tqdm(URLS):
        imgs = getData(img_url, "images", size=size)
        lbl = getData(label_url, "labels", size=size)
        data.extend([imgs, lbl])

    x_train, y_train, x_test, y_test = data

    x_train = x_train.astype("float32")
    x_train /= 255.0

    y_train = to_categorical(y_train)

    x_test = x_test.astype("float32")
    x_test /= 255.0

    y_test = to_categorical(y_test)

    return [x_train, y_train, x_test, y_test]
