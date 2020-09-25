import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from spnflow.utils.statistics import get_statistics
from spnflow.structure.leaf import Gaussian, Multinomial
from spnflow.learning.wrappers import learn_classifier
from spnflow.algorithms.mpe import mpe
from spnflow.io.json import save_json

# Load the dataset
x_data, y_data = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25)
_, n_features = x_data.shape
data_train = np.hstack([x_train, y_train.reshape(-1, 1)])
data_test = np.hstack([x_test, np.full((len(y_test), 1), np.nan)])

# Learn the SPN classifier structure and parameters
distributions = [Gaussian] * n_features + [Multinomial]
spn = learn_classifier(data_train, distributions, min_rows_slice=8)
print(get_statistics(spn))

# Classify samples using maximum at posteriori inference
labels = mpe(spn, data_test)[:, -1]
print('Accuracy: %.1f%%' % (accuracy_score(y_test, labels) * 100))

# Save the SPN model
save_json(spn, 'spn_classifier.json')
