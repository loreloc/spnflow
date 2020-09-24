import warnings
import numpy as np
import scipy.stats as stats
import scipy.sparse as sparse
from joblib import Parallel, delayed
from sklearn import cluster
from sklearn import cross_decomposition
from itertools import combinations


def rdc_cols(data, d=0.3, k=16, s=1.0 / 6.0, f=np.sin, n_jobs=-1):
    """
    Split the features using the RDC (Randomized Dependency Coefficient) method.

    :param data: The data.
    :param d: The threshold value that regulates the independence tests among the features.
    :param k: The size of the latent space.
    :param s: The standard deviation of the gaussian distribution.
    :param f: The non linear function to use.
    :param n_jobs: The number of jobs to use for parallel computation.
    :return: A features partitioning.
    """
    n_samples, n_features = data.shape
    rdc_features = rdc_transform(data, k, s, f)

    pairwise_comparisons = list(combinations(range(n_features), 2))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rdc_values = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(rdc_cca)(i, j, rdc_features) for i, j in pairwise_comparisons
        )

    adj_matrix = np.zeros((n_features, n_features))
    for (i, j), v in zip(pairwise_comparisons, rdc_values):
        if v > d:
            adj_matrix[i, j] = v
            adj_matrix[j, i] = v

    adj_matrix = sparse.csr_matrix(adj_matrix)
    _, clusters = sparse.csgraph.connected_components(adj_matrix, directed=False, return_labels=True)
    return clusters


def rdc_rows(data, n=2, k=16, s=1.0 / 6.0, f=np.sin):
    """
    Split the samples using the RDC (Randomized Dependency Coefficient) method.

    :param data: The data.
    :param n: The number of clusters for KMeans.
    :param k: The size of the latent space.
    :param s: The standard deviation of the gaussian distribution.
    :param f: The non linear function to use.
    :return: A samples partitioning.
    """
    rdc_samples = np.concatenate(rdc_transform(data, k, s, f), axis=1)
    return cluster.KMeans(n_clusters=n).fit_predict(rdc_samples)


def rdc_cca(i, j, features):
    """
    Compute the RDC (Randomized Dependency Coefficient) using CCA (Canonical Correlation Coefficient).

    :param i: The index of the first feature.
    :param j: The index of the second feature.
    :param features: The list of the features.
    :return: The RDC coefficient (the largest canonical correlation coefficient).
    """
    cca = cross_decomposition.CCA(n_components=1, max_iter=100)
    x_cca, y_cca = cca.fit_transform(features[i], features[j])
    return np.corrcoef(x_cca.T, y_cca.T)[0, 1]


def rdc_transform(data, k, s, f):
    """
    Execute the RDC (Randomized Dependency Coefficient) pipeline on some data.

    :param data: The data.
    :param k: The size of the latent space.
    :param s: The standard deviation of the gaussian distribution.
    :param f: The non linear function to use.
    :return: The transformed data.
    """
    n_samples, n_features = data.shape

    features = []
    for i in range(n_features):
        features.append(stats.rankdata(data[:, i], method='max') / n_samples)

    weights = [stats.norm.rvs(0.0, s, size=k) for f in features]
    biases = [stats.norm.rvs(0.0, s) for f in features]

    projected_samples = [np.outer(f, w) + b for f, w, b in zip(features, weights, biases)]
    non_linear_samples = [f(s) for s in projected_samples]

    return non_linear_samples
