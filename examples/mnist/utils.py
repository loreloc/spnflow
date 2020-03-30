import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from examples.mnist.autoencoder import Autoencoder


def plot_fit_history(history, title='Untitled'):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test'])
    plt.show()


def plot_samples(n_rows, n_cols, samples):
    _, axes = plt.subplots(n_rows, n_cols)
    axes = [ax for axs in axes for ax in axs]
    for ax, sample in zip(axes, samples):
        ax.set_axis_off()
        ax.imshow(sample, cmap='gray')
    plt.show()


def dataset_preprocess(train_data, test_data):
    (x_train, y_train), (x_test, y_test) = train_data, test_data
    (n_train_samples, img_w, img_h) = x_train.shape
    (n_test_samples, _, _) = x_test.shape
    n_features = img_w * img_h
    x_train = x_train.reshape(n_train_samples, n_features)
    x_test = x_test.reshape(n_test_samples, n_features)
    y_train = y_train.reshape(n_train_samples, 1)
    y_test = y_test.reshape(n_test_samples, 1)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)


def load_dataset():
    train_data, test_data = tf.keras.datasets.mnist.load_data()
    return dataset_preprocess(train_data, test_data)


"""
(i)   (1 4 7)     -> Vertical Stroke
(ii)  (0 6 8 9)   -> A Circle
(iii) (2 3 5 8 9) -> Left Curvy Stroke
(iv)  (5 6)       -> Right Curvy Stroke
(v)   (7 2 3 4 5) -> Horizontal Stroke
(vi)  (3 8)       -> Double Curve Stroke
"""
def symbolic_info(x):
    if x == 0:
        return [0, 1, 0, 0, 0, 0]
    elif x == 1:
        return [1, 0, 0, 0, 0, 0]
    elif x == 2:
        return [0, 0, 1, 0, 1, 0]
    elif x == 3:
        return [0, 0, 1, 0, 1, 1]
    elif x == 4:
        return [1, 0, 0, 0, 1, 0]
    elif x == 5:
        return [0, 0, 1, 1, 1, 0]
    elif x == 6:
        return [0, 1, 0, 1, 0, 0]
    elif x == 7:
        return [1, 0, 0, 0, 1, 0]
    elif x == 8:
        return [0, 1, 1, 0, 0, 1]
    elif x == 9:
        return [0, 1, 1, 0, 0, 0]


def load_extended_dataset():
    # Load the MNIST dataset and append the symbolic information
    (x_train, y_train), (x_test, y_test) = load_dataset()
    s_train = np.array(list(map(symbolic_info, y_train)))
    s_test = np.array(list(map(symbolic_info, y_test)))
    return (x_train, s_train, y_train), (x_test, s_test, y_test)


def build_autoencoder(encoder_fp, decoder_fp):
    # Load the MNIST Digits dataset
    (x_train, _), (x_test, _) = load_dataset()
    n_features = x_train.shape[1]

    # Build and fit the Autoencoder
    autoencoder = Autoencoder(encoder_name='enc', decoder_name='dec', autoencoder_name='ae')
    autoencoder.build(n_features, 16, (256, 128, 64))
    autoencoder.compile()
    history = autoencoder.fit(x_train, x_train, epochs=200, validation_data=(x_test, x_test))

    # Save the Autoencoder
    autoencoder.save(encoder_fp=encoder_fp, decoder_fp=decoder_fp)

    return history
