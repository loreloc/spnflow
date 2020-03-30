from sklearn import cluster


def kmeans(data, k=2):
    return cluster.KMeans(n_clusters=k).fit_predict(data)
