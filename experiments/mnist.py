import gzip
import pickle
import numpy as np
import tensorflow as tf
from spnflow.model.flow import AutoregressiveRatSpn
from experiments.utils import log_loss

ALPHA = 1e-6

def load_mnist_dataset(rand_state):
    # Load the dataset
    file = gzip.open('datasets/mnist/mnist.pkl.gz', 'rb')
    data_train, data_val, data_test = pickle.load(file, encoding='latin1')
    file.close()
    data_train = data_train[0]
    data_val= data_val[0]
    data_test = data_test[0]

    # Dequantize the dataset
    data_train = data_train + rand_state.rand(*data_train.shape) / 256.0
    data_val = data_val + rand_state.rand(*data_val.shape) / 256.0
    data_test = data_test + rand_state.rand(*data_test.shape) / 256.0

    # Logit transform
    data_train = ALPHA + (1.0 - 2.0 * ALPHA) * data_train
    data_val = ALPHA + (1.0 - 2.0 * ALPHA) * data_val
    data_test = ALPHA + (1.0 - 2.0 * ALPHA) * data_test
    data_train = np.log(data_train / (1.0 - data_train))
    data_val = np.log(data_val / (1.0 - data_val))
    data_test = np.log(data_test / (1.0 - data_test))
    data_train = data_train.astype('float32')
    data_val = data_val.astype('float32')
    data_test = data_test.astype('float32')

    return data_train, data_val, data_test


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
        epochs=2, batch_size=64,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=20)]
    )

    # Compute the test set mean log likelihood
    y_pred = model.predict(data_test)
    mu_log_likelihood = np.mean(y_pred)
    sigma_log_likelihood = np.std(y_pred) / np.sqrt(data_test.shape[0])

    # Compute the bits per pixel metric
    z = tf.math.sigmoid(data_test)
    bits_correction = data_train.shape[1] * np.log(1.0 - 2.0 * ALPHA) - np.sum(np.log(z) + np.log(1.0 - z), axis=1)
    y_pred_bits = y_pred + bits_correction
    mu_bits_log_likelihood = np.mean(y_pred_bits)
    sigma_bits_log_likelihood = np.std(y_pred_bits) / np.sqrt(data_test.shape[0])
    dlog2 = data_train.shape[1] * np.log(2)
    mu_bpp = -mu_bits_log_likelihood / dlog2 + 8.0
    sigma_bpp = sigma_bits_log_likelihood / dlog2

    # Save the results to file
    with open('results/mnist.txt', 'w') as file:
        file.write('MNIST;\n')
        file.write('Avg. Log-Likelihood: ' + str(mu_log_likelihood) + '\n')
        file.write('Std. Log-Likelihood: ' + str(sigma_log_likelihood) + '\n')
        file.write('Avg. Bits Per Pixel: ' + str(mu_bpp) + '\n')
        file.write('Std. Bits Per Pixel: ' + str(sigma_bpp) + '\n')
