import numpy as np
from sklearn.datasets import load_breast_cancer

from deeprob.spn.structure.leaf import Gaussian
from deeprob.spn.models.sklearn import SPNEstimator, SPNClassifier

# Load the dataset and set the features distributions
data, _ = load_breast_cancer(return_X_y=True)
_, n_features = data.shape
distributions = [Gaussian] * n_features

# Fit the density estimator
clf = SPNEstimator(
    distributions,
    learn_leaf='mle',
    split_rows='kmeans',
    split_cols='rdc',
    min_rows_slice=64
)
clf.fit(data)

# Compute the re-substitution mean LL and two standard deviations
score = clf.score(data)
print('Train - Mean LL: {} - Stddev LL: {}'.format(score['mean_ll'], score['stddev_ll']))

# Compute the mean LL and two standard deviations from sampled data
data = clf.sample(n=100)
score = clf.score(data)
print('Sampling - Mean LL: {} - Stddev LL: {}'.format(score['mean_ll'], score['stddev_ll']))

# =====================================================================================================================

# Load the dataset and set the features distributions
data, target = load_breast_cancer(return_X_y=True)
_, n_features = data.shape
distributions = [Gaussian] * n_features

# Fit the classifier
clf = SPNClassifier(
    distributions,
    learn_leaf='mle',
    split_rows='kmeans',
    split_cols='rdc',
    min_rows_slice=64
)
clf.fit(data, target)

# Compute the re-substitution accuracy score
print('Train - Accuracy: {}'.format(clf.score(data, target)))

# Sample from data from the conditional distribution
classes = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1])
samples = clf.sample(y=classes)
print('Sampling - Accuracy: {}'.format(clf.score(samples[:, :-1], classes)))
