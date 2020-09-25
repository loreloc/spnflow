import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from spnflow.utils.statistics import get_statistics
from spnflow.structure.leaf import Gaussian
from spnflow.learning.wrappers import learn_estimator
from spnflow.algorithms.sampling import sample
from spnflow.io.json import save_json

# Load the dataset
data, _ = make_moons(n_samples=10000, noise=0.05)
_, n_features = data.shape

# Learn the SPN density estimation structure and parameters
# Minimum number of samples required for clustering is 1024 and the number of clusters is 8
distributions = [Gaussian] * n_features
spn = learn_estimator(data, distributions, min_rows_slice=1024, split_rows_kwargs={'n': 8})
print(get_statistics(spn))

# Sample some values
samples = np.full((1000, 2), np.nan)
samples = sample(spn, samples)

# Plot the samples
plt.scatter(samples[:, 0], samples[:, 1], s=2)
plt.show()

# Save the SPN model
save_json(spn, 'spn_estimator.json')
