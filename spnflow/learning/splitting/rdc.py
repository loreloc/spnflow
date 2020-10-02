import warnings
import numpy as np
import scipy.stats as stats
import scipy.sparse as sparse

from sklearn import cluster
from sklearn import cross_decomposition
from itertools import combinations
from spnflow.structure.leaf import LeafType
from spnflow.utils.data import ohe_data


def rdc_cols(data, distributions, domains, d=0.3, k=16, s=1.0 / 6.0, f=np.sin):
    """
    Split the features using the RDC (Randomized Dependency Coefficient) method.

    :param data: The data.
    :param distributions: The data distributions.
    :param domains: The data domains.
    :param d: The threshold value that regulates the independence tests among the features.
    :param k: The size of the latent space.
    :param s: The standard deviation of the gaussian distribution.
    :param f: The non linear function to use.
    :return: A features partitioning.
    """
    n_samples, n_features = data.shape
    rdc_features = rdc_transform(data, distributions, domains, k, s, f)
    pairwise_comparisons = list(combinations(range(n_features), 2))

    adj_matrix = np.zeros((n_features, n_features), dtype=np.int)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i, j in pairwise_comparisons:
            rdc = rdc_cca(i, j, rdc_features)
            if rdc > d:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

    adj_matrix = sparse.csr_matrix(adj_matrix)
    _, clusters = sparse.csgraph.connected_components(adj_matrix, directed=False, return_labels=True)
    return clusters


def rdc_rows(data, distributions, domains, n=2, k=16, s=1.0 / 6.0, f=np.sin):
    """
    Split the samples using the RDC (Randomized Dependency Coefficient) method.

    :param data: The data.
    :param distributions: The data distributions.
    :param domains: The data domains.
    :param n: The number of clusters for KMeans.
    :param k: The size of the latent space.
    :param s: The standard deviation of the gaussian distribution.
    :param f: The non linear function to use.
    :return: A samples partitioning.
    """
    rdc_samples = np.concatenate(rdc_transform(data, distributions, domains, k, s, f), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return cluster.KMeans(n_clusters=n).fit_predict(rdc_samples)


def rdc_cca(i, j, features):
    """
    Compute the RDC (Randomized Dependency Coefficient) using CCA (Canonical Correlation Coefficient).

    :param i: The index of the first feature.
    :param j: The index of the second feature.
    :param features: The list of the features.
    :return: The RDC coefficient (the largest canonical correlation coefficient).
    """
    cca = cross_decomposition.CCA(n_components=1, max_iter=128)
    x_cca, y_cca = cca.fit_transform(features[i], features[j])
    return np.corrcoef(x_cca.T, y_cca.T)[0, 1]


def rdc_transform(data, distributions, domains, k, s, f):
    """
    Execute the RDC (Randomized Dependency Coefficient) pipeline on some data.

    :param data: The data.
    :param distributions: The data distributions.
    :param domains: The data domains.
    :param k: The size of the latent space.
    :param s: The standard deviation of the gaussian distribution.
    :param f: The non linear function to use.
    :return: The transformed data.
    """
    # Empirical Cumulative Distribution Function
    def ecdf(x):
        return stats.rankdata(x, method='max') / len(x)

    features = []
    for i, dist in enumerate(distributions):
        if dist.LEAF_TYPE == LeafType.DISCRETE:
            feature_matrix = ohe_data(data[:, i], domains[i])
        else:
            feature_matrix = np.expand_dims(data[:, i], axis=-1)
        features.append(np.apply_along_axis(ecdf, 0, feature_matrix))

    weights = [stats.norm.rvs(size=(f.shape[1], f.shape[1] * k)) for f in features]
    projected_samples = [(s / f.shape[1]) * np.dot(f, w) for f, w in zip(features, weights)]
    non_linear_samples = [f(x) for x in projected_samples]

    return non_linear_samples
