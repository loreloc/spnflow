import numpy as np
import scipy.stats as stats
import scipy.sparse as sparse
from sklearn import cross_decomposition
from itertools import combinations


def split_cols_clusters(data, clusters, scope):
    slices = []
    scopes = []
    np_scope = np.asarray(scope)
    unique_clusters = np.unique(clusters)
    for c in unique_clusters:
        cols = clusters == c
        local_data = data[:, cols]
        slices.append(local_data)
        scopes.append(np_scope[cols].tolist())
    return slices, scopes


def get_split_cols_method(split_cols):
    if split_cols == 'rdc':
        return rdc
    else:
        raise NotImplementedError("Unknow split rows method called " + split_cols)


def rdc(data, threshold):
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


def rdc_transform(data, k=20, s=1.0/6.0, nl_func=np.sin):
    n_samples, n_features = data.shape

    features = []
    #for i in range(n_features):
    #    col = data[:, i]
    #    if issubclass(col.dtype.type, np.integer):
    #        features.append(ohe_data(col).reshape(-1, 1))
    #    else:
    #        features.append(col.reshape(-1, 1))
    features = [data[:, i].reshape(-1, 1) for i in range(n_features)]

    features = [empirical_copula_transform(f) for f in features]

    gaussian_samples = [stats.norm.rvs(0.0, s, size=(2, k)) for i in range(n_features)]
    projected_samples = [np.dot(c, w) for c, w in zip(features, gaussian_samples)]
    non_linear_samples = [nl_func(c) for c in projected_samples]

    ones_column = np.ones((n_samples, 1))
    return [np.concatenate((phi, ones_column), axis=1) for phi in non_linear_samples]


def rdc_cca(i, j, features):
    cca = cross_decomposition.CCA(n_components=1, max_iter=100)
    x_cca, y_cca = cca.fit_transform(features[i], features[j])
    corr = np.corrcoef(x_cca.T, y_cca.T)
    return corr[0, 1]


def ohe_data(data):
    n_samples, _ = data.shape
    domain = np.unique(data)
    ohe = np.zeros((n_samples, len(domain)))
    ohe[data == domain] = 1
    return ohe


def empirical_copula_transform(data):
    def ecdf(x):
        return stats.rankdata(x, method='max') / len(x)
    n_samples, _ = data.shape
    ones_column = np.ones((n_samples, 1))
    return np.concatenate((np.apply_along_axis(ecdf, 0, data), ones_column), axis=1)
