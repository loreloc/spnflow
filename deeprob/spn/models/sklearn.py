import numpy as np
from scipy.special import log_softmax
from sklearn.base import BaseEstimator, DensityMixin, ClassifierMixin

from deeprob.spn.structure.leaf import Bernoulli, Categorical
from deeprob.spn.learning.wrappers import learn_estimator, learn_classifier
from deeprob.spn.algorithms.inference import log_likelihood, mpe
from deeprob.spn.algorithms.sampling import sample


class SPNEstimator(BaseEstimator, DensityMixin):
    """
    Scikit-learn density estimator model for Sum Product Networks.
    """
    def __init__(
            self,
            distributions,
            domains=None,
            learn_leaf='mle',
            learn_leaf_kwargs=None,
            split_rows='gmm',
            split_cols='rdc',
            split_rows_kwargs=None,
            split_cols_kwargs=None,
            min_rows_slice=256,
            min_cols_slice=2,
            random_state=None,
            verbose=True
    ):
        """
        Initialize the density estimator.

        :param distributions: The base distributions associated to each feature.
        :param domains: The domains associated to each feature. Use None to automatically determine the domains.
        :param learn_leaf: The method to use to learn a distribution leaf node
                           (it can be either 'mle', 'isotonic' or 'cltree').
        :param learn_leaf_kwargs: The parameters of the learn leaf method.
        :param split_rows: The rows splitting method (it can be 'kmeans', 'gmm', 'rdc' or 'random').
        :param split_cols: The columns splitting method (it can be 'gvs', 'rdc' or 'random').
        :param split_rows_kwargs: The parameters of the rows splitting method.
        :param split_cols_kwargs: The parameters of the cols splitting method.
        :param min_rows_slice: The minimum number of samples required to split horizontally.
        :param min_cols_slice: The minimum number of features required to split vertically.
        :param random_state: The random state. It can be either None, a seed integer or a Numpy RandomState.
        :param verbose: Whether to enable verbose mode.
        """
        super(SPNEstimator, self).__init__()
        self.distributions = distributions
        self.domains = domains
        self.learn_leaf = learn_leaf
        self.learn_leaf_kwargs = learn_leaf_kwargs
        self.split_rows = split_rows
        self.split_cols = split_cols
        self.split_rows_kwargs = split_rows_kwargs
        self.split_cols_kwargs = split_cols_kwargs
        self.min_rows_slice = min_rows_slice
        self.min_cols_slice = min_cols_slice
        self.random_state = random_state
        self.verbose = verbose
        self.spn_ = None
        self.n_features_ = 0

    def fit(self, X, y=None):
        """
        Fit the SPN density estimator.

        :param X: The training data.
        :param y: Ignored, only for scikit-learn API convention.
        :return: Itself.
        """
        self.spn_ = learn_estimator(
            data=X, distributions=self.distributions, domains=self.domains,
            learn_leaf=self.learn_leaf, learn_leaf_kwargs=self.learn_leaf_kwargs,
            split_rows=self.split_rows, split_cols=self.split_cols,
            split_rows_kwargs=self.split_rows_kwargs, split_cols_kwargs=self.split_cols_kwargs,
            min_rows_slice=self.min_rows_slice, min_cols_slice=self.min_cols_slice,
            random_state=self.random_state, verbose=self.verbose
        )
        _, self.n_features_ = X.shape
        return self

    def predict_log_proba(self, X):
        """
        Predict using the SPN density estimator, i.e. compute the log-likelihood.

        :param X: The inputs. They can be marginalized using NaNs.
        :return: The log-likelihood of the inputs.
        """
        return log_likelihood(self.spn_, X)

    def mpe(self, X):
        """
        Predict the un-observed variable by maximum at posterior estimation (MPE).

        :param X: The inputs having some NaN values.
        :return: The MPE assignment to un-observed variables.
        """
        return mpe(self.spn_, X)

    def sample(self, n=None, X=None):
        """
        Sample from the modeled distribution.

        :param n: The number of samples. It must be None if X is not None. If None, n=1 is assumed.
        :param X: Data used for conditional sampling. It can be None for full sampling.
        :return: The samples.
        """
        assert n is None or X is None, "Only one between n and X can be specified"
        if X is not None:
            # Check for conditional sampling
            return sample(self.spn_, X)
        else:
            # Full sampling
            n = 1 if n is None else n
            x = np.tile(np.nan, [n, self.n_features_])
            return sample(self.spn_, x)

    def score(self, X, y=None):
        """
        Return the mean log-likelihood and two standard deviations on the given test data.

        :param X: The inputs. They can be marginalized using NaNs.
        :param y: Ignored. Specified only for scikit-learn API compatibility.
        :return: A dictionary consisting of two keys "mean_ll" and "stddev_ll",
                 representing respectively the mean log-likelihood and two standard deviations.
        """
        ll = self.predict_log_proba(X)
        mean_ll = np.mean(ll)
        stddev_ll = np.std(ll)
        return {
            'mean_ll': mean_ll,
            'stddev_ll': 2.0 * stddev_ll / np.sqrt(len(X))
        }


class SPNClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn classifier model for Sum Product Networks.
    """
    def __init__(
            self,
            distributions,
            domains=None,
            learn_leaf='mle',
            learn_leaf_kwargs=None,
            split_rows='gmm',
            split_cols='rdc',
            split_rows_kwargs=None,
            split_cols_kwargs=None,
            min_rows_slice=256,
            min_cols_slice=2,
            random_state=None,
            verbose=True
    ):
        """
        Initialize the density estimator.

        :param distributions: The base distributions associated to each feature.
        :param domains: The domains associated to each feature. Use None to automatically determine the domains.
        :param learn_leaf: The method to use to learn a distribution leaf node
                           (it can be 'mle', 'isotonic' or 'cltree').
        :param learn_leaf_kwargs: The parameters of the learn leaf method.
        :param split_rows: The rows splitting method (it can be 'kmeans', 'gmm', 'rdc' or 'random').
        :param split_cols: The columns splitting method (it can be 'gvs', 'rdc' or 'random').
        :param split_rows_kwargs: The parameters of the rows splitting method.
        :param split_cols_kwargs: The parameters of the cols splitting method.
        :param min_rows_slice: The minimum number of samples required to split horizontally.
        :param min_cols_slice: The minimum number of features required to split vertically.
        :param random_state: The random state. It can be either None, a seed integer or a Numpy RandomState.
        :param verbose: Whether to enable verbose mode.
        """
        super(SPNClassifier, self).__init__()
        self.distributions = distributions
        self.domains = domains
        self.learn_leaf = learn_leaf
        self.learn_leaf_kwargs = learn_leaf_kwargs
        self.split_rows = split_rows
        self.split_cols = split_cols
        self.split_rows_kwargs = split_rows_kwargs
        self.split_cols_kwargs = split_cols_kwargs
        self.min_rows_slice = min_rows_slice
        self.min_cols_slice = min_cols_slice
        self.random_state = random_state
        self.verbose = verbose
        self.spn_ = None
        self.n_features_ = 0
        self.n_classes_ = 0

    def fit(self, X, y):
        """
        Fit the SPN density estimator.

        :param X: The training data.
        :param y: The data labels.
        :return: Itself.
        """
        # Build the training data, consisting of labels
        y = np.expand_dims(y, axis=1)
        data = np.hstack([X, y])

        # Constructs the list of distributions
        n_classes = len(np.unique(y))
        if n_classes == 2:
            # Use bernoulli for binary classification
            distributions = self.distributions + [Bernoulli]
        else:
            # otherwise, use a categorical distribution
            distributions = self.distributions + [Categorical]

        self.spn_ = learn_classifier(
            data=data, distributions=distributions, domains=self.domains,
            learn_leaf=self.learn_leaf, learn_leaf_kwargs=self.learn_leaf_kwargs,
            split_rows=self.split_rows, split_cols=self.split_cols,
            split_rows_kwargs=self.split_rows_kwargs, split_cols_kwargs=self.split_cols_kwargs,
            min_rows_slice=self.min_rows_slice, min_cols_slice=self.min_cols_slice,
            random_state=self.random_state, verbose=self.verbose
        )
        _, self.n_features_ = X.shape
        self.n_classes_ = n_classes
        return self

    def predict(self, X):
        """
        Predict using the SPN classifier.

        :param X: The inputs. They can be marginalized using NaNs.
        :return: The predicted classes.
        """
        # Build the testing data, having X as features assignments and NaNs for labels
        data = np.hstack([X, np.tile(np.nan, [len(X), 1])])

        # Make classification using maximum at posterior estimation (MPE)
        mpe_data = mpe(self.spn_, data)

        # Return the classifications for each sample
        return mpe_data[:, -1]

    def predict_proba(self, X):
        """
        Predict using the SPN classifier, using probabilities.

        :param X: The inputs. They can be marginalized using NaNs.
        :return: The prediction probabilities for each class.
        """
        return np.exp(self.predict_log_proba(X))

    def predict_log_proba(self, X):
        """
        Predict using the SPN classifier, using log-probabilities.

        :param X: The inputs. They can be marginalized using NaNs.
        :return: The prediction log-probabilities for each class.
        """
        # Build the testing data, having X as features assignments and NaNs for labels
        data = np.hstack([X, np.tile(np.nan, [len(X), 1])])

        # Make probabilistic classification by computing the log-likelihoods at sub-class SPN
        _, ls = log_likelihood(self.spn_, data, return_results=True)

        # Collect the predicted class probabilities
        class_ids = [c.id for c in self.spn_.children]
        class_ll = np.log(self.spn_.weights) + ls[class_ids]
        return log_softmax(class_ll, axis=1)

    def sample(self, n=None, y=None):
        """
        Sample from the modeled conditional distribution.

        :param n: The number of samples. It must be None if y is not None. If None, n=1 is assumed.
        :param y: Labels used for conditional sampling. It can be None for un-conditional sampling.
        :return: The samples.
        """
        assert n is None or y is None, "Only one between n and y can be specified"
        if y is not None:
            # Check for conditional sampling
            y = np.expand_dims(y, axis=1)
            x = np.hstack([np.tile(np.nan, [len(y), self.n_features_]), y])
            return sample(self.spn_, x)
        else:
            # Full sampling
            n = 1 if n is None else n
            x = np.tile(np.nan, [n, self.n_features_ + 1])
            return sample(self.spn_, x)
