import numpy as np
import scipy.stats as stats
import scipy.sparse as sparse
from scipy.special import logsumexp

from deeprob.spn.structure.leaf import Leaf, LeafType, Bernoulli
from deeprob.spn.structure.node import Sum, Mul, assign_ids


class TreeNode:
    """A simple class to model a node of a tree."""
    def __init__(self, node_id, parent=None):
        """
        Initialize a binary CLT.

        :param node_id: The ID of the node.
        :param parent: The parent node.
        """
        self.node_id = node_id
        self.set_parent(parent)
        self.children = []

    def get_node_id(self):
        """
        Get the ID of the node.
        
        :return: The ID of the node.
        """
        return self.node_id

    def get_parent(self):
        """
        Get the parent node.

        :return: The parent node, None if the node has no parent.
        """
        return self.parent

    def get_children(self):
        """
        Get the children list of the node.

        :return: The children list of the node.
        """
        return self.children

    def set_parent(self, parent):
        """
        Set the parent node and update its children list.
        """
        if parent is not None:
            self.parent = parent
            self.parent.children.append(self)

    def is_leaf(self):
        """
        Check whether the node is leaf.

        :return: True if the node is leaf, False otherwise.
        """
        return len(self.children) == 0


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
        assert root is None or root in self.scope, "The root variable must be in scope"
        self.bfs = bfs
        self.tree = tree

        # Initialize the parameters
        if params is not None:
            self.params = np.array(params, dtype=np.float32)
        else:
            self.params = None

        # Initialize the root variable
        if root is not None:
            self.root = self.scope.index(root)
        else:
            self.root = None

    def fit(self, data, domain, alpha=0.1, random_state=None, **kwargs):
        """
        Fit the distribution parameters given the domain and some training data.

        :param data: The training data.
        :param domain: The domain of the distribution leaf.
        :param alpha: Laplace smoothing factor.
        :param random_state: The random state. It can be either None, a seed integer or a Numpy RandomState.
        :param kwargs: Optional parameters.
        """
        n_samples, n_features = data.shape
        assert n_features == len(self.scope), "Number of features and leaf scope mismatch"

        # Initialize the random state
        if random_state is None:
            random_state = np.random.RandomState()
        elif type(random_state) == int:
            random_state = np.random.RandomState(random_state)
        elif not isinstance(random_state, np.random.RandomState):
            raise ValueError("The random state must be either None, a seed integer or a Numpy RandomState")

        # Choose a root variable randomly, if not specified
        if self.root is None:
            self.root = random_state.choice(len(self.scope))

        # Make sure to work with float32
        data = data.astype(np.float32)

        # Estimate the priors and joints probabilities
        priors, joints = self.__estimate_priors_joints(data, alpha=alpha)

        # Compute the mutual information
        mutual_info = self.__compute_mutual_information(priors, joints)

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
        joints[:, :, 0, 0] = n_samples - counts_cols - counts_rows + counts_ones
        joints[:, :, 0, 1] = counts_cols - counts_ones
        joints[:, :, 1, 0] = counts_rows - counts_ones
        joints[:, :, 1, 1] = counts_ones
        joints = (joints + alpha) / (n_samples + 4 * alpha)
        return priors, joints

    @staticmethod
    def __compute_mutual_information(priors, joints):
        """
        Compute the mutual information between each features, given priors and joints distributions.

        :param priors: The priors probability distributions.
        :param joints: The joints probability distributions.
        :return: The mutual information between each pair of features (as a symmetric matrix).
        """
        # Compute the mutual information
        outers = np.multiply.outer(priors, priors).transpose([0, 2, 1, 3])
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

        # Compute the conditional probabilities (by einsum operation)
        params[features] = np.einsum('ikl,il->ilk', joints[features, parents], np.reciprocal(priors[parents]))
        return params

    def __message_passing(self, x):
        """
        Compute the messages passed from the leaves to the root node.

        :param x: The input data.
        :return: The messages array.
        """
        # Let's proceed bottom-up
        n_features = len(x)
        messages = np.zeros(shape=(n_features, 2), dtype=np.float32)
        for j in reversed(self.bfs[1:]):
            # If non-observed value then factor marginalize that variable
            if np.isnan(x[j]):
                messages[self.tree[j]] += logsumexp(self.params[j] + messages[j], axis=1)
            else:
                obs_value = int(x[j])
                messages[self.tree[j]] += self.params[j, :, obs_value] + messages[j, obs_value]
        return messages

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
            # Compute the final likelihood considering the root node
            # Note that self.params[self.root, 1] = self.params[self.root, 0], since it is unconditioned
            messages = self.__message_passing(x[i])
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
            # Do the message passing
            messages = self.__message_passing(x[i])

            # Do MPE at the root feature, if necessary
            if np.isnan(z[i, self.root]):
                probs = self.params[self.root, 0, :] + messages[self.root, 0]
                z[i, self.root] = np.argmax(probs)

            # Do MPE at the other features, by using the accumulated messages
            for j in self.bfs[1:]:
                if np.isnan(z[i, j]):
                    obs_parent_value = int(z[i, self.tree[j]])
                    probs = self.params[j, obs_parent_value] + messages[j, obs_parent_value]
                    z[i, j] = np.argmax(probs)
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
            # Do the message passing
            messages = self.__message_passing(x[i])

            # Sample the root feature, if necessary
            if np.isnan(z[i, self.root]):
                prob = np.exp(self.params[self.root, 0, 1] + messages[self.root, 0])
                z[i, self.root] = stats.bernoulli.rvs(prob)

            # Sample the other features, by using the accumulated messages
            for j in self.bfs[1:]:
                if np.isnan(z[i, j]):
                    obs_parent_value = int(z[i, self.tree[j]])
                    prob = np.exp(self.params[j, obs_parent_value, 1] + messages[j, obs_parent_value])
                    z[i, j] = stats.bernoulli.rvs(prob)
        return z

    def to_pc(self):
        """
        Convert a Chow-Liu Tree into a smooth, deterministic and structured-decomposable PC

        :return: A smooth, deterministic and structured-decomposable PC.
        """

        # Build tree structure
        tree_nodes = {node_id: TreeNode(node_id) for node_id in range(len(self.scope))}
        [tree_nodes[i].set_parent(tree_nodes[self.tree[i]]) for i in range(len(self.scope)) if i != self.root]

        neg_buffer, pos_buffer = [], []
        nodes_stack = [tree_nodes[self.root]]
        last_node_visited = None
        # Post-Order exploration
        while nodes_stack:
            node = nodes_stack[-1]
            if node.is_leaf() or (last_node_visited in node.children):
                leaves = [Bernoulli(p=0.0, scope=[self.scope[node.node_id]]),
                          Bernoulli(p=1.0, scope=[self.scope[node.node_id]])]
                if not node.is_leaf():
                    neg_prod = Mul(children=[leaves[0]] + neg_buffer[-len(node.children):])
                    pos_prod = Mul(children=[leaves[1]] + pos_buffer[-len(node.children):])
                    del neg_buffer[-len(node.children):]
                    del pos_buffer[-len(node.children):]
                    sum_children = [neg_prod, pos_prod]
                else:
                    sum_children = leaves
                weights = np.exp(self.params[node.node_id])
                # Re-normalize the weights, because there can be FP32 approximation errors
                weights /= np.expand_dims(np.sum(weights, axis=1), axis=-1)
                neg_buffer.append(
                    Sum(children=sum_children, weights=weights[0]))
                pos_buffer.append(
                    Sum(children=sum_children, weights=weights[1]))
                last_node_visited = nodes_stack.pop()
            else:
                nodes_stack.extend(node.children)
        # equivalently, pos = neg_buffer[0]
        pc = pos_buffer[0]
        return assign_ids(pc)

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
