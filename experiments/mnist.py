import numpy as np
import tensorflow as tf
from spnflow.model.flow import AutoregressiveRatSpn
from experiments.utils import log_loss


def load_mnist_dataset(rand_state):
    # Load the dataset
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], 784))
    x_test = x_test.reshape((x_test.shape[0], 784))
    rand_state.shuffle(x_train)

    # Dequantize the dataset
    x_train = (x_train + rand_state.rand(*x_train.shape)) / 256.0
    x_test = (x_test + rand_state.rand(*x_test.shape)) / 256.0

    # Logit transform
    alpha = 1e-6
    x_train = alpha + (1.0 - 2.0 * alpha) * x_train
    x_test = alpha + (1.0 - 2.0 * alpha) * x_test
    x_train = np.log(x_train / (1.0 - x_train))
    x_test = np.log(x_test / (1.0 - x_test))
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Split the dataset in train and validation dataset
    n_val = int(0.1 * x_train.shape[0])
    x_val = x_train[-n_val:]
    x_train = x_train[0:-n_val]

    return x_train, x_val, x_test


if __name__ == '__main__':
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the dataset
    data_train, data_val, data_test = load_mnist_dataset(rand_state)

    # Build the model
    model = AutoregressiveRatSpn(
        depth=3,
        n_batch=8,
        n_sum=8,
        n_repetitions=32,
        n_mafs=5,
        hidden_units=[1024],
        activation='relu',
        regularization=1e-6,
        rand_state=rand_state
    )

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=log_loss)

    # Fit the model
    model.fit(
        x=data_train,
        y=np.zeros((data_train.shape[0], 0), dtype=np.float32),
        validation_data=(data_val, np.zeros((data_val.shape[0], 0), dtype=np.float32)),
        epochs=500, batch_size=64,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=20)]
    )

    # Compute the test set mean log likelihood
    y_pred = model.predict(data_test)
    mu_log_likelihood = np.mean(y_pred)
    sigma_log_likelihood = np.std(y_pred) / np.sqrt(data_test.shape[0])

    # Save the results to file
    with open('results/mnist.txt', 'w') as file:
        file.write('MNIST;\n')
        file.write('Avg. Log-Likelihood: ' + str(mu_log_likelihood) + '\n')
        file.write('Std. Log-Likelihood: ' + str(sigma_log_likelihood) + '\n')
