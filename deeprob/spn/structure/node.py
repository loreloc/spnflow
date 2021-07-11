import abc
import numpy as np

from collections import deque, defaultdict
from scipy.special import softmax, logsumexp


class Node(abc.ABC):
    """SPN node base class."""
    def __init__(self, children, scope):
        """
        Initialize a node given the children list and its scope.

        :param children: A list of nodes.
        :param scope: The scope.
        """
        assert sorted(scope) == scope, 'The scope of a node must be sorted'
        self.id = None
        self.children = children
        self.scope = scope

    @abc.abstractmethod
    def likelihood(self, x):
        """
        Compute the likelihood of the node given some input.

        :param x: The inputs.
        :return: The resulting likelihood.
        """
        pass

    @abc.abstractmethod
    def log_likelihood(self, x):
        """
        Compute the logarithmic likelihood of the node given some input.

        :param x: The inputs.
        :return: The resulting log likelihood.
        """
        pass


class Sum(Node):
    """The sum node class."""
    def __init__(self, weights, children=None, scope=None):
        """
        Initialize a sum node given a list of children and their weights and a scope.

        :param weights: The weights.
        :param children: A list of nodes.
        :param scope: The scope.
        """
        if children is None:
            children = []
        if scope is None:
            scope = []
        if len(scope) == 0 and len(children) > 0:
            scope = children[0].scope
        super().__init__(children, scope)
        if isinstance(weights, list):
            weights = np.array(weights, dtype=np.float32)
        self.weights = weights

    def em_init(self, random_state):
        """
        Random initialize the node's parameters for Expectation-Maximization (EM).

        :param random_state: The random state.
        """
        w = 1e-1 * random_state.randn(len(self.children))
        self.weights = softmax(w).astype(np.float32)

    def em_step(self, stats):
        """
        Compute the parameters after an EM step.

        :param stats: The sufficient statistics of each sample.
        :return: A dictionary of new parameters.
        """
        unnorm_weights = self.weights * np.sum(stats, axis=1) + np.finfo(np.float32).eps
        weights = unnorm_weights / np.sum(unnorm_weights)
        return {'weights': weights}

    def likelihood(self, x):
        """
        Compute the likelihood of the sum node given some input.

        :param x: The inputs.
        :return: The resulting likelihood.
        """
        return np.expand_dims(np.dot(x, self.weights), axis=1)

    def log_likelihood(self, x):
        """
        Compute the logarithmic likelihood of the node given some input.

        :param x: The inputs.
        :return: The resulting log likelihood.
        """
        return logsumexp(x, b=self.weights, axis=1, keepdims=True)


class Mul(Node):
    """The multiplication node class."""
    def __init__(self, children=None, scope=None):
        """
        Initialize a sum node given a list of children and their weights and a scope.

        :param children: A list of nodes.
        :param scope: The scope.
        """
        if children is None:
            children = []
        if scope is None:
            scope = []
        if len(scope) == 0 and len(children) > 0:
            scope = list(sorted(sum([c.scope for c in children], [])))
        super().__init__(children, scope)

    def likelihood(self, x):
        """
        Compute the likelihood of the multiplication node given some input.

        :param x: The inputs.
        :return: The resulting likelihood.
        """
        return np.prod(x, axis=1, keepdims=True)

    def log_likelihood(self, x):
        """
        Compute the logarithmic likelihood of the multiplication node given some input.

        :param x: The inputs.
        :return: The resulting log likelihood.
        """
        return np.sum(x, axis=1, keepdims=True)


def assign_ids(root):
    """
    Assign the ids to the nodes of a SPN.

    :param root: The root of the SPN.
    :return: The same SPN with each node having modified ids.
    """
    next_id = 0

    def assign_id(node):
        nonlocal next_id
        node.id = next_id
        next_id += 1

    bfs(root, assign_id)
    return root


def bfs(root, func):
    """
    Breadth First Search (BFS) for SPN.
    For each node execute a given function.

    :param root: The root of the SPN.
    :param func: The function to evaluate for each node.
    """
    seen, queue = {root}, deque([root])
    while queue:
        node = queue.popleft()
        func(node)
        for c in node.children:
            if c not in seen:
                seen.add(c)
                queue.append(c)


def dfs_post_order(root, func):
    """
    Depth First Search (DFS) Post-Order for SPN.
    For each node execute a given function.

    :param root: The root of the SPN.
    :param func: The function to evaluate for each node.
    """
    seen, stack = {root}, [root]
    while stack:
        node = stack[-1]
        if set(node.children).issubset(seen):
            func(node)
            stack.pop()
        else:
            for c in node.children:
                if c not in seen:
                    seen.add(c)
                    stack.append(c)


def topological_order(root):
    """
    Compute the Topological Ordering for a SPN, using the Kahn's Algorithm.

    :param root: The root of the SPN.
    :return: A list of nodes that form a topological ordering.
             If the SPN graph is not acyclic, it returns None.
    """
    num_incomings = defaultdict(int)
    num_incomings[root] = 0

    def count_incomings(node):
        for c in node.children:
            num_incomings[c] += 1
    bfs(root, count_incomings)

    ordering = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        ordering.append(node)
        for c in node.children:
            num_incomings[c] -= 1
            if num_incomings[c] == 0:
                queue.append(c)

    if sum(num_incomings.values()) > 0:
        return None
    return ordering
