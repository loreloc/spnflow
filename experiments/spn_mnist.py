import numpy as np
import sklearn as sk

from experiments.datasets import DatasetTransform, load_vision_dataset
from experiments.utils import save_grid_images

from spnflow.utils.statistics import get_statistics
from spnflow.structure.leaf import Gaussian, Multinomial
from spnflow.learning.wrappers import learn_classifier
from spnflow.algorithms.mpe import mpe
from spnflow.algorithms.sampling import sample
from spnflow.io.json import save_json


# Load the MNIST dataset
image_w, image_h = 28, 28
n_classes = 10
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_vision_dataset('datasets', 'mnist', unsupervised=False)

# Preprocess the dataset
transform = DatasetTransform(dequantize=True, standardize=False, flatten=True)
transform.fit(np.row_stack([x_train, x_valid]))
x_train = transform.forward(x_train)
x_valid = transform.forward(x_valid)
x_test = transform.forward(x_test)
x_train = np.row_stack([x_train, x_valid])
y_train = np.hstack([y_train, y_valid])

# Apply PCA for dimensionality reduction
n_components = 40
pca = sk.decomposition.PCA(n_components)
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

# Learn the SPN structure and parameters, as a classifier
distributions = [Gaussian] * n_components + [Multinomial]
data_train = np.column_stack([x_train, y_train])
spn = learn_classifier(
    data_train, distributions,
    learn_leaf='mle',
    split_rows='kmeans',
    split_cols='gvs',
    min_rows_slice=256,
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
data_sample = np.column_stack([nans, classes])
data_sample = sample(spn, data_sample)[:, :-1]
data_images = transform.backward(pca.inverse_transform(data_sample))
data_images = data_images.reshape(n_samples, n_classes, 1, image_w, image_h)
save_grid_images(data_images, 'spn-mnist-samples.png')
