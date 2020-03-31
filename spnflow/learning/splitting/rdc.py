import numpy as np
import scipy.stats as stats
import scipy.sparse as sparse
from sklearn import cross_decomposition
from itertools import combinations


def rdc(data, threshold):
    """
    Split the features using the RDC (Randomized Dependency Coefficient) method.

    :param data: The data.
    :param threshold: The threshold value that regulates the independence tests among the features.
    :return:
    """
    n_samples, n_features = data.shape
    rdc_features = rdc_transform(data)

    pairwise_comparisons = list(combinations(range(n_features), 2))
    rdc_values = [rdc_cca(i, j, rdc_features) for i, j in pairwise_comparisons]

    adj_matrix = np.zeros((n_features, n_features))
    for (i, j), v in zip(pairwise_comparisons, rdc_values):
        adj_matrix[i, j] = v
        adj_matrix[j, i] = v

    adj_matrix[np.isnan(adj_matrix)] = 0
    adj_matrix[adj_matrix <= threshold] = 0
    adj_matrix[adj_matrix > 0] = 1

    adj_matrix = sparse.csr_matrix(adj_matrix)
    _, clusters = sparse.csgraph.connected_components(adj_matrix, directed=False, return_labels=True)
    return clusters


def rdc_cca(i, j, features):
    """
    Compute the RDC (Randomized Dependency Coefficient) using CCA (Canonical Correlation Coefficient).

    :param i: The index of the first feature.
    :param j: The index of the second feature.
    :param features: The list of the features.
    :return: The RDC coefficient (the largest canonical correlation coefficient).
    """
    cca = cross_decomposition.CCA(n_components=1)
    x_cca, y_cca = cca.fit_transform(features[i], features[j])
    return np.corrcoef(x_cca.T, y_cca.T)[0, 1]


def rdc_transform(data, k=20, s=1.0/6.0, nl_func=np.sin):
    """
    Execute the RDC (Randomized Dependency Coefficient) pipeline on some data.

    :param data: The data.
    :param k: The size of the latent space.
    :param s: The standard deviation of the gaussian distribution.
    :param nl_func: The non linear function to use.
    :return: The transformed data.
    """
    n_samples, n_features = data.shape

    features = []
    for i in range(n_features):
        features.append(stats.rankdata(data[:, i], method='max') / n_samples)

    weights = [stats.norm.rvs(0.0, s, size=k) for f in features]
    bases = [stats.norm.rvs(0.0, s, size=k) for f in features]

    projected_samples = [np.add(np.outer(f, w), b) for f, w, b in zip(features, weights, bases)]

    non_linear_samples = [nl_func(s) for s in projected_samples]

    return non_linear_samples
