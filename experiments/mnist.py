import os
import gzip
import pickle
import numpy as np

ALPHA = 1e-6
IMG_SIZE = 28


def load_mnist_dataset(rand_state, use_dequantize=True, logit_space=True, flatten=True):
    # Load the dataset
    filepath = os.path.join(os.environ['DATAPATH'], 'datasets/mnist/mnist.pkl.gz')
    file = gzip.open(filepath, 'rb')
    data_train, data_val, data_test = pickle.load(file, encoding='latin1')
    file.close()
    data_train = data_train[0]
    data_val = data_val[0]
    data_test = data_test[0]

    # Dequantize the dataset
    if use_dequantize:
        data_train = dequantize(data_train, rand_state)
        data_val = dequantize(data_val, rand_state)
        data_test = dequantize(data_test, rand_state)

    # Logit transform
    if logit_space:
        data_train = logit(data_train)
        data_val = logit(data_val)
        data_test = logit(data_test)

    # Check the flatten flag
    if not flatten:
        data_train = np.reshape(data_train, (-1, 1, IMG_SIZE, IMG_SIZE))
        data_val = np.reshape(data_val, (-1, 1, IMG_SIZE, IMG_SIZE))
        data_test = np.reshape(data_test, (-1, 1, IMG_SIZE, IMG_SIZE))

    data_train = data_train.astype('float32')
    data_val = data_val.astype('float32')
    data_test = data_test.astype('float32')

    return data_train, data_val, data_test


def logit(data):
    data = ALPHA + (1.0 - 2.0 * ALPHA) * data
    return np.log(data / (1.0 - data))


def delogit(data):
    x = 1.0 / (1.0 + np.exp(-data))
    return (x - ALPHA) / (1.0 - 2.0 * ALPHA)


def dequantize(data, rand_state):
    return data + rand_state.rand(*data.shape) / 256.0


def plot(ax, sample):
    ax.imshow(np.reshape(sample, (IMG_SIZE, IMG_SIZE)), vmin=0.0, vmax=1.0, cmap='gray', interpolation='nearest')
