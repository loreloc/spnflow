import torch
import torchvision
import numpy as np
import sklearn as sk

from spnflow.utils.data import DataStandardizer
from spnflow.utils.statistics import get_statistics
from spnflow.structure.leaf import Gaussian, Multinomial
from spnflow.learning.wrappers import learn_classifier
from spnflow.algorithms.mpe import mpe
from spnflow.algorithms.sampling import sample
from spnflow.io.json import save_json

image_w, image_h = 28, 28
n_classes = 10

# Load the MNIST dataset
data_train = torchvision.datasets.MNIST('datasets', train=True, transform=None, download=True)
data_test = torchvision.datasets.MNIST('datasets', train=False, transform=None, download=True)
x_train, y_train = data_train.data.numpy(), data_train.targets.numpy()
x_test, y_test = data_test.data.numpy(), data_test.targets.numpy()
x_train = np.reshape(x_train, [len(x_train), -1])
x_test = np.reshape(x_test, [len(x_test), -1])

# Apply PCA for dimensionality reduction
n_components = 32
pca = sk.decomposition.PCA(n_components)
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

# Preprocess the dataset
transform = DataStandardizer()
transform.fit(x_train)
x_train = transform.forward(x_train)
x_test = transform.forward(x_test)

# Learn the SPN structure and parameters, as a classifier
distributions = [Gaussian] * n_components + [Multinomial]
data_train = np.column_stack([x_train, y_train])
spn = learn_classifier(
    data_train,
    distributions,
    learn_leaf='mle',
    split_rows='kmeans',
    split_cols='gvs',
    min_rows_slice=200,
    split_rows_kwargs={'n': 2},
    split_cols_kwargs={'p': 1.0}
)

# Print some statistics
print(get_statistics(spn))

# Save the model to a JSON file
save_json(spn, 'spn_mnist.json')

# Make some predictions on the test set classes
nans = np.tile(np.nan, [len(x_test), 1])
data_test = np.column_stack([x_test, nans])
y_pred = mpe(spn, data_test)[:, -1]

# Plot a summary of the classification
print(sk.metrics.classification_report(y_test, y_pred))

# Make some sampling for each class
n_samples = 10
nans = np.tile(np.nan, [n_samples * n_classes, n_components])
classes = np.tile(np.arange(n_classes), [1, n_samples]).T
data_samples = np.column_stack([nans, classes])
data_samples = sample(spn, data_samples)[:, :-1]

# Transform back the data and plot the samples
data_images = pca.inverse_transform(transform.backward(data_samples))
data_images = data_images.reshape(n_samples * n_classes, 1, image_h, image_w)
data_images = data_images / 255.0
torchvision.utils.save_image(torch.tensor(data_images), 'samples_spn_mnist.png', nrow=n_samples, padding=0)
