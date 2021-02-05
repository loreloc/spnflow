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
    for i, j in pairwise_comparisons:
        rdc = rdc_svd(i, j, rdc_features)
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
    return cluster.KMeans(n_clusters=n).fit_predict(rdc_samples)


def rdc_svd(i, j, features):
    """
    Compute the RDC (Randomized Dependency Coefficient) using SVD (Singular Value Decomposition).

    :param i: The index of the first feature.
    :param j: The index of the second feature.
    :param features: The list of the features.
    :return: The RDC coefficient (the largest canonical correlation coefficient).
    """
    svd = cross_decomposition.PLSSVD(n_components=1)
    svd.fit(features[i], features[j])
    x, y = svd.transform(features[i], features[j])
    return np.corrcoef(x.T, y.T)[0, 1]


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

    biases = [stats.norm.rvs(0.0, s, size=(1, k)) for f in features]
    weights = [stats.norm.rvs(0.0, s, size=(f.shape[1], k)) for f in features]
    projected_samples = [np.dot(f, w) + b for f, w, b in zip(features, weights, biases)]
    non_linear_samples = [f(x) for x in projected_samples]

    return non_linear_samples
