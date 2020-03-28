import os
import numpy as np
import sklearn as sk
import tensorflow as tf
from spnflow.utils.validity import assert_is_valid
from spnflow.utils.statistics import get_statistics
from spnflow.learning.wrappers import learn_classifier
from spnflow.structure.leaf import Gaussian, Multinomial
from spnflow.algorithms.inference import likelihood, log_likelihood
from spnflow.algorithms.mpe import mpe
from spnflow.algorithms.sampling import sample
from spnflow.examples.mnist.utils import build_autoencoder, plot_fit_history, load_dataset, plot_samples

# Disable some warnings
import warnings
warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')

if __name__ == '__main__':
    # Set the random state
    seed = 42
    rnd = np.random.RandomState(seed)

    # Set some constants
    n_class = 10
    (img_w, img_h) = img_shape = (28, 28)

    # Load the MNIST digits dataset
    (x_train, y_train), (x_test, y_test) = load_dataset()

    # Load the encoder and decoder
    if not os.path.isfile('encoder.h5') or not os.path.isfile('decoder.h5'):
        history = build_autoencoder('encoder.h5', 'decoder.h5')
        plot_fit_history(history, 'MNIST Digits Autoencoder Loss')
    encoder = tf.keras.models.load_model('encoder.h5')
    decoder = tf.keras.models.load_model('decoder.h5')


    # Code the entire MNIST digits dataset
    x_train = encoder.predict(x_train)
    x_test = encoder.predict(x_test)
    n_samples, n_features = x_train.shape

    # Preprocess the encoded MNIST digits dataset
    train_data = np.concatenate((x_train, y_train), axis=1)
    test_data = np.concatenate((x_test, y_test), axis=1)

    # Learn the SPN classifier
    distributions = [Gaussian] * n_features + [Multinomial]
    domains = [[0.0, 1.0]] * n_features + [n_class]
    spn = learn_classifier(train_data, distributions, domains, class_idx=n_features, min_rows_slice=2048)
    assert_is_valid(spn)
    print("SPN Statistics: " + str(get_statistics(spn)))

    # Evaluate some metrics
    ll = log_likelihood(spn, test_data)
    print("Log Likelihood: " + str(np.mean(ll)))
    test_data[:, n_features] = np.nan
    y_hat = mpe(spn, test_data)[:, n_features]
    print("Accuracy Score: " + str(sk.metrics.accuracy_score(y_test, y_hat)))

    # Do some Sampling
    queries = []
    n_instances = 10
    # Sample all the digits
    for k in range(n_instances):
        for i in range(n_class):
            queries.append([np.nan] * n_features + [i])
    # Execute the queries
    samples = sample(spn, queries)[:, :n_features]
    samples = decoder.predict(samples)
    samples = samples.reshape(len(queries), img_w, img_h)

    # Plot the samples
    n_rows = n_instances
    n_cols = n_class
    plot_samples(n_rows, n_cols, samples)
