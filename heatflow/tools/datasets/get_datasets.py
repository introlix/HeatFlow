import os
import requests
import gzip
import numpy as np

def parseFileName(url: str) -> str:
    return url.split("/")[-1]

def parseDataset(f_path: str, size: int):
    reader = gzip.open(f_path, "rb")
    reader.read(16)  # first 16 bits contains the magic number and the dimension of the images

    data_buffer = reader.read((28 * 28 * int(size)))

    data = np.frombuffer(data_buffer, dtype=np.uint8).astype(np.float32)
    data = data.reshape(size, 784)

    return data

def parseLabels(f_path: str, size: int):
    reader = gzip.open(f_path, "rb")
    reader.read(8)
    labels = []

    buf = reader.read(size)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64).reshape(size, 1)

    return labels

def getData(url: str, data_type: str, size: int):
    """
    Returns the data as numpy array

    Parameter
    ---------
    Arg: url (str)
        Link to the dataset
    Arg: data (str)
        "images" or "labels"
    Arg: size (int)
        size of the dataset to read
    """

    # Ensure the 'data' directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    f_path = os.path.join('data', parseFileName(url))

    if not os.path.exists(f_path):
        response = requests.get(url)

        with open(f_path, "wb") as file:
            file.write(response.content)

    if data_type == "images":
        data = parseDataset(f_path, size=size)
    elif data_type == "labels":
        data = parseLabels(f_path, size=size)

    return data
