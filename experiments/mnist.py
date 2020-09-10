import os
import torchvision
import numpy as np
from experiments.utils import logit, delogit, dequantize

IMG_SIZE = 28


def load_mnist_dataset(rand_state, flatten=True, apply_dequant=True, apply_logit=True):
    # Load the dataset
    datasets_root = os.path.join(os.environ['DATAPATH'], 'datasets')
    data_train = torchvision.datasets.MNIST(datasets_root, train=True, download=True)
    data_test = torchvision.datasets.MNIST(datasets_root, train=False, download=True)
    data_train = data_train.data.numpy()
    data_test = data_test.data.numpy()
    rand_state.shuffle(data_train)

    # Check the flatten flag
    if flatten:
        data_train = np.reshape(data_train, (len(data_train), -1))
        data_test = np.reshape(data_test, (len(data_test), -1))
    else:
        data_train = np.reshape(data_train, (len(data_train), 1, IMG_SIZE, IMG_SIZE))
        data_test = np.reshape(data_test, (len(data_test), 1, IMG_SIZE, IMG_SIZE))

    # Split the train dataset in train and validation set
    n_val = int(0.1 * len(data_train))
    data_val = data_train[-n_val:]
    data_train = data_train[0:-n_val]

    if apply_dequant:
        # Dequantize the dataset
        data_train = dequantize(data_train, rand_state)
        data_val = dequantize(data_val, rand_state)
        data_test = dequantize(data_test, rand_state)

    # Normalize the dataset
    data_train /= 255.0
    data_val /= 255.0
    data_test /= 255.0

    if apply_logit:
        # Logit transform
        data_train = logit(data_train)
        data_val = logit(data_val)
        data_test = logit(data_test)

    data_train = data_train.astype('float32')
    data_val = data_val.astype('float32')
    data_test = data_test.astype('float32')

    return data_train, data_val, data_test


def to_images(samples):
    samples = np.clip(delogit(samples), 0.0, 1.0)
    return np.reshape(samples, (-1, 1, IMG_SIZE, IMG_SIZE))
