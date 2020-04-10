import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from spnflow.model import build_spn


def plot_fit_history(history, metric='loss', title='Untitled'):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test'])
    plt.show()


def get_loss_function(kind='mixed', lam=1.0):
    @tf.function
    def log_loss(y_true, y_pred):
        return tf.math.negative(y_pred)

    @tf.function
    def cross_entropy(y_true, y_pred):
        return tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)

    @tf.function
    def mixed_loss(y_true, y_pred):
        return lam * cross_entropy(y_true, y_pred) + (1 - lam) * log_loss(y_true, y_pred)

    if kind == 'mixed':
        return mixed_loss
    elif kind == 'log_loss':
        return log_loss
    elif kind == 'cross_entropy':
        return cross_entropy
    else:
        raise NotImplementedError("Loss function kind not implemented")


if __name__ == '__main__':
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess the MNIST dataset
    n_classes = 10
    n_features = 784
    scaler = StandardScaler()
    x_train = np.reshape(x_train, (x_train.shape[0], n_features)).astype(np.float32)
    x_test = np.reshape(x_test, (x_test.shape[0], n_features)).astype(np.float32)
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    y_train = tf.keras.utils.to_categorical(y_train, n_classes)
    y_test = tf.keras.utils.to_categorical(y_test, n_classes)

    # Build the RAT-SPN model
    depth = 2
    n_sum = 10
    n_dists = 2
    n_reps = 10
    spn = build_spn(n_features, n_classes, depth, n_sum, n_dists, n_reps)

    # Print some summary
    spn.summary()

    # Compile the model
    loss_fn = get_loss_function(kind='cross_entropy')
    spn.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    # Fit the model
    history = spn.fit(x_train, y_train, batch_size=256, epochs=100, validation_data=(x_test, y_test))

    # Plot the train history
    plot_fit_history(history, metric='loss', title='Loss')
    plot_fit_history(history, metric='accuracy', title='Accuracy')
