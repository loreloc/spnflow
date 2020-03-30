from sklearn import mixture


def gmm(data, k=2):
    return mixture.GaussianMixture(n_components=k).fit_predict(data)
