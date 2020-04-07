import numpy as np
import tensorflow as tf
from spnflow.model import build_spn
from spnflow.region import RegionGraph
import matplotlib.pyplot as plt


def plot_fit_history(history, metric='loss', title='Untitled'):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test'])
    plt.show()


def get_loss_function(lam=1.0):
    def log_loss(y_true, y_pred):
        return tf.math.negative(tf.reduce_mean(y_pred))

    def cross_entropy(y_true, y_pred):
        return tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)

    def mixed_loss(y_true, y_pred):
        return lam * cross_entropy(y_true, y_pred) + (1 - lam) * log_loss(y_true, y_pred)

    return mixed_loss


if __name__ == '__main__':
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess the MNIST dataset
    n_classes = 10
    n_features = 784
    x_train = np.reshape(x_train, (x_train.shape[0], n_features)).astype(np.float32) / 255.0
    x_test = np.reshape(x_test, (x_test.shape[0], n_features)).astype(np.float32) / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, n_classes)
    y_test = tf.keras.utils.to_categorical(y_test, n_classes)

    # Generate the region graph's layers
    depth = 2
    n_repetitions = 10
    region_graph = RegionGraph(list(range(n_features)))
    for i in range(n_repetitions):
        region_graph.random_split(depth)
    region_graph.make_layers()
    layers = region_graph.layers()

    # Construct the RAT-SPN model
    n_sum = 10
    n_distributions = 2
    spn = build_spn(n_classes, n_sum, n_distributions, layers)

    # Print some summary
    spn.build()
    spn.summary()

    # Compile the model
    spn.compile(optimizer='adam', loss=get_loss_function(lam=1.0), metrics=['accuracy'])

    # Fit the model
    history = spn.fit(x_train, y_train, batch_size=256, epochs=100, validation_data=(x_test, y_test))

    # Plot the train history
    plot_fit_history(history, metric='loss', title='Loss')
    plot_fit_history(history, metric='accuracy', title='Accuracy')
