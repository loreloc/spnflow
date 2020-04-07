import numpy as np
import tensorflow as tf
from spnflow.model import build_spn
from spnflow.region import RegionGraph


if __name__ == '__main__':
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess the MNIST dataset
    n_classes = 10
    n_features = 784
    x_train = np.reshape(x_train, (x_train.shape[0], n_features)).astype(np.float32)
    x_test = np.reshape(x_test, (x_test.shape[0], n_features)).astype(np.float32)
    y_train = tf.keras.utils.to_categorical(y_train, n_classes)
    y_test = tf.keras.utils.to_categorical(y_test, n_classes)

    # Generate the region graph's layers
    depth = 2
    n_repetitions = 2
    region_graph = RegionGraph(list(range(n_features)))
    for i in range(n_repetitions):
        region_graph.random_split(depth)
    region_graph.make_layers()
    layers = region_graph.layers()

    # Construct the RAT-SPN model
    n_sum = 10
    n_distributions = 16
    spn = build_spn(n_classes, n_sum, n_distributions, layers)

    # Print some summary
    spn.build()
    spn.summary()

    # Compile the model
    spn.compile(optimizer='adam', loss=tf.nn.softmax_cross_entropy_with_logits, metrics=['accuracy'])

    # Fit the model
    history = spn.fit(x_train, y_train, batch_size=256, epochs=100)
