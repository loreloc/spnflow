import os
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from spnflow.wrappers import build_autoregressive_flow_spn


RESULTS_DIR = "results/"

EPOCHS = 1

BATCH_SIZE = 256

HYPER_PARAMETERS = {
    'depth': [2, 3, 4],
    'hidden_units': [[32, 32], [64, 64]],
    'regularization': [1e-4, 1e-5, 1e-6],
    'n_sum': [10, 20, 30],
    'n_reps': [20, 30, 40],
    'dropout': [1.0, 0.8]
}


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


def build_hyper_parameters_space():
    """
    Build the hyper-parameters list.
    """
    return (dict(zip(HYPER_PARAMETERS.keys(), values)) for values in itertools.product(*HYPER_PARAMETERS.values()))


def benchmark_classification():
    """
    Run the benchmark of the classification task.
    """
    # Make sure results directory exists
    os.mkdir(RESULTS_DIR)

    # Load the MNIST dataset
    x_train, y_train, x_test, y_test = load_mnist_dataset()
    n_features = x_train.shape[1]
    n_classes = y_train.shape[1]

    # Get the loss function to use
    loss_fn = get_loss_function(kind='cross_entropy')

    # Build the hyper-parameters space
    hp_space = build_hyper_parameters_space()

    # Create the hyper-parameters space and results data frame
    hp_space_df = pd.DataFrame(columns=list(HYPER_PARAMETERS.keys()) + ['val_loss', 'val_accuracy'])

    for idx, hp in enumerate(hp_space):
        # Build the model
        spn = build_autoregressive_flow_spn(n_features, n_classes, **hp)

        # Compile the model
        spn.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

        # Fit the model
        history = spn.fit(x_train, y_train, beatch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test))

        # Make directory
        model_path = os.path.join(RESULTS_DIR, str(idx).zfill(4))
        os.mkdir(model_path)

        # Save the history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(model_path, 'history.csv'))

        # Save the validation loss and accuracy at the end of the training
        hp_space_df.loc[idx, hp.keys()] = hp
        hp_space_df.loc[idx, 'val_loss'] = history.history['val_loss'][-1]
        hp_space_df.loc[idx, 'val_accuracy'] = history.history['val_accuracy'][-1]

    # Save the hyper-parameters space and results to file
    hp_space_df.to_csv(os.path.join(RESULTS_DIR, 'hpspace.csv'))
