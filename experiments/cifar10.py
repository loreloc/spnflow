import pickle
import numpy as np

ALPHA = 0.05
IMG_SIZE = 32
IMG_DEPTH = 3


def load_cifar10_dataset(rand_state):
    path = 'datasets/cifar10/'

    # Load the train dataset
    x = []
    for i in range(1, 6):
        file = open(path + 'data_batch_' + str(i), 'rb')
        data = pickle.load(file, encoding='latin1')
        x.append(data['data'])
        file.close()
    data = np.concatenate(x, axis=0)
    rand_state.shuffle(data)

    # Split the dataset in train and validation set
    n_val = int(0.1 * data.shape[0])
    data_val = data[-n_val:]
    data_train = data[0:-n_val]

    # Load the test dataset
    file = open(path + 'test_batch', 'rb')
    data = pickle.load(file, encoding='latin1')
    data_test = data['data']
    file.close()

    # Preprocess the dataset
    data_train = dequantize(data_train, rand_state)
    data_val = dequantize(data_val, rand_state)
    data_test = dequantize(data_test, rand_state)
    data_train = logit(data_train)
    data_val = logit(data_val)
    data_test = logit(data_test)

    # Augment the train set using horizontal flip
    data_train = augment_flip(data_train)

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
    return (data + rand_state.rand(*data.shape)) / 256.0


def augment_flip(data):
    data = np.reshape(data, (data.shape[0], IMG_SIZE, IMG_SIZE, IMG_DEPTH))
    flip_data = data[:, :, ::-1]
    data = np.reshape(data, (data.shape[0], -1))
    flip_data = np.reshape(flip_data, (data.shape[0], -1))
    return np.vstack([data, flip_data])


def plot(ax, sample):
    ax.imshow(np.reshape(sample, (IMG_SIZE, IMG_SIZE, IMG_DEPTH)), interpolation='nearest')
