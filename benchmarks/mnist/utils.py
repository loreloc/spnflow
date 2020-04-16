import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# The results directory
RESULTS_DIR = "results"

# The number of training epochs
EPOCHS = 100

# The batch size
BATCH_SIZE = 256


def load_mnist_dataset(standardize=True, features_select=True):
    """
    Load the MNIST dataset.

    :param standardize: Apply standardization to the dataset.
    :param features_select: Remove uninformative features from the dataset.
    """
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess the MNIST dataset
    n_classes = 10
    n_features = 784
    x_train = np.reshape(x_train, (x_train.shape[0], n_features)).astype(np.float32)
    x_test = np.reshape(x_test, (x_test.shape[0], n_features)).astype(np.float32)
    y_train = tf.keras.utils.to_categorical(y_train, n_classes)
    y_test = tf.keras.utils.to_categorical(y_test, n_classes)

    if standardize:
        # Apply standardization
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    if features_select:
        # Remove uninformative features
        filter = VarianceThreshold(1e-1)
        filter.fit(x_train)
        x_train = filter.transform(x_train)
        x_test = filter.transform(x_test)

    return x_train, y_train, x_test, y_test


def get_loss_function(kind='mixed', n_features=None, lam=None):
    """
    Get the loss function to use.

    :param kind: The kind. It  can be 'log_loss', 'cross_entropy' and 'mixed'.
    :param n_features: The number of features (needed if kind='log_loss').
    :param lam: The "trade-off" parameter (needed if kind='mixed').
    :return: The loss function.
    """
    @tf.function
    def log_loss(y_true, y_pred):
        return tf.math.negative(tf.math.reduce_mean(y_pred)) / n_features

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
        raise NotImplementedError("Unknown loss function")
