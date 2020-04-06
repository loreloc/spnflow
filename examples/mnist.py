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
    x_train = np.reshape(x_train, (x_train.shape[0], n_features)).astype(np.float32) / 255.0
    x_test = np.reshape(x_test, (x_test.shape[0], n_features)).astype(np.float32) / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, n_classes)
    y_test = tf.keras.utils.to_categorical(y_test, n_classes)

    # Generate the region graph's layers
    depth = 4
    n_repetitions = 2
    region_graph = RegionGraph(list(range(n_features)))
    for i in range(n_repetitions):
        region_graph.random_split(depth)
    region_graph.make_layers()
    layers = region_graph.layers()

    # Construct the RAT-SPN model
    n_sum = 2
    n_distributions = 2
    spn = build_spn(n_classes, n_sum, n_distributions, layers)

    # Compile the model
    spn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print some summary
    spn.summary()

    # Fit the model
    history = spn.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))
