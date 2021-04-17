import numpy as np
import scipy.stats as stats
import scipy.sparse as sparse
from scipy.special import logsumexp

from spnflow.structure.leaf import Leaf, LeafType


class BinaryCLTree(Leaf):
    """
    Binary Chow-Liu tree (CLT) multi-binary leaf node.

    Thanks to Gennaro Gala (https://github.com/gengala) for his implementation.
    """

    LEAF_TYPE = LeafType.DISCRETE
    
    def __init__(self, scope, root=None, bfs=None, tree=None, params=None):
        """
        Initialize a binary CLT.

        :param scope: The scope of the leaf.
        """
        super(BinaryCLTree, self).__init__(scope)
        self.root = root
        self.bfs = bfs
        self.tree = tree
        if params is not None:
            self.params = np.array(params, dtype=np.float32)
        else:
            self.params = None

    def fit(self, data, domain, alpha=0.1, **kwargs):
        """
        Fit the distribution parameters given the domain and some training data.

        :param data: The training data.
        :param domain: The domain of the distribution leaf.
        :param alpha: Laplace smoothing factor.
        :param kwargs: Optional parameters.
        """
        n_samples, n_features = data.shape
        assert n_features == len(self.scope), "Number of features and leaf scope mismatch"

        # Make sure to work with float32
        data = data.astype(np.float32)

        # Estimate the priors and joints probabilities
        priors, joints = self.__estimate_priors_joints(data, alpha=alpha)

        # Compute the mutual information
        mutual_info = self.__compute_mutual_information(priors, joints)

        # Choose a starting root randomly, used to compute the maximum spanning tree
        self.root = np.random.choice(n_features)

        # Compute the CLT structure by getting the maximum spanning tree of the mutual-information graph
        # Note adding one to the mutual information, because the graph must be connected
        ms_tree = sparse.csgraph.minimum_spanning_tree(-(mutual_info + 1.0))
        bfs, tree = sparse.csgraph.breadth_first_order(
            ms_tree, directed=False, i_start=self.root, return_predecessors=True
        )
        bfs[0] = self.root
        tree[self.root] = -1
        self.bfs = bfs.tolist()
        self.tree = tree.tolist()

        # Compute the CLT parameters (in log-space), using the joints and priors probabilities
        params = self.__compute_clt_parameters(self.tree, priors, joints)
        self.params = np.log(params)

    @staticmethod
    def __estimate_priors_joints(data, alpha=0.1):
        """
        Estimate both priors and joints probability distributions from data.

        :param data: The binary data.
        :param alpha: Laplace smoothing factor.
        :return: A pair of priors and joints distributions.
                 Note that priors[i, k] := P(X_i=k).
                 Note that joints[i, j, k, l] := P(X_i=k, X_j=l).
        """
        # Compute the counts
        n_samples, n_features = data.shape
        counts_ones = np.dot(data.T, data)
        counts_features = np.diag(counts_ones)
        counts_cols = counts_features * np.ones_like(counts_ones)
        counts_rows = np.transpose(counts_cols)

        # Compute the prior probabilities
        priors = np.empty(shape=(n_features, 2), dtype=np.float32)
        priors[:, 1] = (counts_features + 2 * alpha) / (n_samples + 4 * alpha)
        priors[:, 0] = 1.0 - priors[:, 1]

        # Compute the joints probabilities
        joints = np.empty(shape=(n_features, n_features, 2, 2), dtype=np.float32)
        joints[:, :, 0, 0] = n_samples - counts_cols - counts_rows + counts_ones + alpha
        joints[:, :, 0, 1] = counts_cols - counts_ones + alpha
        joints[:, :, 1, 0] = counts_rows - counts_ones + alpha
        joints[:, :, 1, 1] = counts_ones + alpha
        joints /= n_samples + 4 * alpha

        return priors, joints

    @staticmethod
    def __compute_mutual_information(priors, joints):
        """
        Compute the mutual information between each features, given priors and joints distributions.

        :param priors: The priors probability distributions.
        :param joints: The joints probability distributions.
        :return: The mutual information between each pair of features (as a symmetric matrix).
        """
        n_features = len(priors)
        outers = np.empty((n_features, n_features, 2, 2), dtype=np.float32)
        outers[:, :, 0, 0] = np.outer(priors[:, 0], priors[:, 0])
        outers[:, :, 0, 1] = np.outer(priors[:, 0], priors[:, 1])
        outers[:, :, 1, 0] = np.outer(priors[:, 1], priors[:, 0])
        outers[:, :, 1, 1] = np.outer(priors[:, 1], priors[:, 1])

        # Compute the mutual information
        mutual_info = np.sum(joints * (np.log(joints) - np.log(outers)), axis=(2, 3))
        np.fill_diagonal(mutual_info, 0.0)
        return mutual_info

    @staticmethod
    def __compute_clt_parameters(tree, priors, joints):
        """
        Compute the parameters of the CLTree given the tree structure and the priors and joints distributions.

        :param tree: The tree structure, i.e. a list of predecessors in a tree structure.
        :param priors: The priors distributions.
        :param joints: The joints distributions.
        :return: The conditional probability tables (CPTs) in a tensorized form.
                 Note that params[i, l, k] = P(X_i=k | Pa(X_i)=l).
                 A special case is made for the root distribution which is not conditioned.
                 Note that params[root, :, k] = P(X_root=k).
        """
        n_features = len(priors)
        params = np.empty(shape=(n_features, 2, 2), dtype=np.float32)
        root_id = tree.index(-1)
        features = list(range(n_features))
        features.remove(root_id)
        parents = tree.copy()
        parents.pop(root_id)

        # Compute the parameters of the root node
        # (note the conditionally independence respect to a dummy variable)
        params[root_id] = priors[root_id]

        # Compute the conditional probabilities
        # Use the transposed joints to subsequently vectorize the divisions
        transposed_joints = np.transpose(joints[features, parents], axes=(0, 2, 1))
        params[features] = transposed_joints / np.expand_dims(priors[parents], axis=2)

        return params

    def likelihood(self, x):
        """
        Compute the likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting likelihood.
        """
        return np.exp(self.log_likelihood(x))

    def log_likelihood(self, x):
        """
        Compute the logarithmic likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting log likelihood.
        """
        n_samples, n_features = x.shape
        ll = np.empty(n_samples, dtype=np.float32)

        # Build the mask of samples with missing values (used for marginalization)
        mask = np.any(np.isnan(x), axis=1)

        # Vectorization of full-evidence inference
        vs = np.arange(n_features)
        z = x[~mask]
        z_cond = z[:, self.tree].astype(np.int64)
        z_vals = z[:, vs].astype(np.int64)
        ll[~mask] = np.sum(self.params[vs, z_cond, z_vals], axis=1)

        # Un-vectorized implementation of marginalized inference
        marg_indices = np.where(mask)[0]
        for i in marg_indices:
            messages = np.zeros(shape=(n_features, 2), dtype=np.float32)

            # Let's proceed bottom-up
            for j in reversed(self.bfs[1:]):
                # If non-observed value then factor marginalize that variable
                if np.isnan(x[i, j]):
                    messages[self.tree[j]] += logsumexp(self.params[j] + messages[j], axis=1)
                else:
                    obs_value = int(x[i, j])
                    messages[self.tree[j]] += self.params[j, :, obs_value] + messages[j, obs_value]

            # Compute the final likelihood considering the root node
            # Note that self.params[self.root, 1] = self.params[self.root], since it is unconditioned
            if np.isnan(x[i, self.root]):
                ll[i] = logsumexp(self.params[self.root, 0] + messages[self.root])
            else:
                obs_value = int(x[i, self.root])
                ll[i] = self.params[self.root, 0, obs_value] + messages[self.root, obs_value]
        return np.expand_dims(ll, axis=1)

    def mpe(self, x):
        """
        Compute the maximum at posteriori values.

        :return: The distribution's maximum at posteriori values.
        """
        z = np.copy(x)
        n_samples, n_features = x.shape

        # Un-vectorized implementation of MPE inference
        for i in range(n_samples):
            messages = np.zeros(shape=(n_features, 2), dtype=np.float32)
            states = np.empty(shape=(n_features, 2), dtype=np.int64)
            # Let's proceed bottom-up
            for j in reversed(self.bfs[1:]):
                # If non-observed value then factor marginalize that variable
                if np.isnan(z[i, j]):
                    # Consider all the possible combinations of maximum likelihood estimation given a parent value
                    parent_mpe_params = self.params[j] + messages[j]
                    parent_mpe_indices = np.argmax(parent_mpe_params, axis=1)
                    states[j] = parent_mpe_indices
                    messages[self.tree[j]] += np.diag(parent_mpe_params[:, parent_mpe_indices])
                else:
                    # Set the states at prior
                    obs_value = int(x[i, j])
                    states[j] = obs_value
                    messages[self.tree[j]] += self.params[j, :, obs_value] + messages[j, obs_value]

            # Compute the final maximum likelihood estimation considering the root node
            # Note that self.params[self.root, 0] = self.params[self.root, 1], since it is unconditioned
            if np.isnan(z[i, self.root]):
                root_params = self.params[self.root, 0] + messages[self.root]
                z[i, self.root] = np.argmax(root_params)

            # Proceed top-down doing assignments from the obtained states
            for j in self.bfs[1:]:
                if np.isnan(z[i, j]):
                    obs_parent_value = int(z[i, self.tree[j]])
                    z[i, j] = states[j, obs_parent_value]
        return z

    def sample(self, x):
        """
        Sample from the leaf distribution.

        :param x: The samples with possible NaN values.
        :return: The completed samples.
        """
        z = np.copy(x)
        n_samples, n_features = x.shape

        # Un-vectorized implementation of conditional sampling
        for i in range(n_samples):
            messages = np.zeros(shape=(n_features, 2), dtype=np.float32)
            states = np.empty(shape=(n_features, 2), dtype=np.int64)
            # Let's proceed bottom-up
            for j in reversed(self.bfs[1:]):
                # If non-observed value then factor marginalize that variable
                if np.isnan(z[i, j]):
                    # Consider all the possible combinations of probabilities given a parent value
                    parent_probs = self.params[j] + messages[j]
                    values = stats.bernoulli.rvs(np.exp(parent_probs[:, 1]))
                    states[j] = values
                    messages[self.tree[j]] += np.diag(parent_probs[:, values])
                else:
                    # Set the states at prior
                    obs_value = int(x[i, j])
                    states[j] = obs_value
                    messages[self.tree[j]] += self.params[j, :, obs_value] + messages[j, obs_value]

            # Compute the final sample considering the root node
            # Note that self.params[self.root, 0] = self.params[self.root, 1], since it is unconditioned
            if np.isnan(z[i, self.root]):
                root_prob = self.params[self.root, 0, 1] + messages[self.root, 1]
                z[i, self.root] = stats.bernoulli.rvs(np.exp(root_prob))

            # Proceed top-down doing assignments from the obtained states
            for j in self.bfs[1:]:
                if np.isnan(z[i, j]):
                    obs_parent_value = int(z[i, self.tree[j]])
                    z[i, j] = states[j, obs_parent_value]
        return z

    def params_count(self):
        """
        Get the number of parameters of the distribution leaf.

        :return: The number of parameters.
        """
        return 1 + len(self.bfs) + len(self.tree) + self.params.size

    def params_dict(self):
        """
        Get a dictionary representation of the distribution parameters.

        :return: A dictionary containing the distribution parameters.
        """
        return {
            'root': self.root,
            'bfs': self.bfs,
            'tree': self.tree,
            'params': self.params.tolist()
        }
