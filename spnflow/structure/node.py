import numpy as np
from scipy.special import logsumexp
from collections import deque


class Node:
    """SPN node base class."""
    def __init__(self, children, scope):
        """
        Initialize a node given the children list and its scope.

        :param children: A list of nodes.
        :param scope: The scope.
        """
        self.id = None
        self.children = children
        self.scope = scope

    def likelihood(self, x):
        """
        Compute the likelihood of the node given some input.

        :param x: The inputs.
        :return: The resulting likelihood.
        """
        pass

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

        :param weights: The weights list.
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
        self.weights = weights

    def likelihood(self, x):
        """
        Compute the likelihood of the sum node given some input.

        :param x: The inputs.
        :return: The resulting likelihood.
        """
        return np.dot(x, self.weights)

    def log_likelihood(self, x):
        """
        Compute the logarithmic likelihood of the node given some input.

        :param x: The inputs.
        :return: The resulting log likelihood.
        """
        return logsumexp(x, b=self.weights, axis=1)


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
            scope = sum([c.scope for c in children], [])
        super().__init__(children, scope)

    def likelihood(self, x):
        """
        Compute the likelihood of the multiplication node given some input.

        :param x: The inputs.
        :return: The resulting likelihood.
        """
        return np.prod(x, axis=1)

    def log_likelihood(self, x):
        """
        Compute the logarithmic likelihood of the multiplication node given some input.

        :param x: The inputs.
        :return: The resulting log likelihood.
        """
        return np.sum(x, axis=1)


def assign_ids(root):
    """
    Assign the ids to the nodes of a SPN.

    :param root: The root of the SPN.
    :return: The same SPN with each node having modified ids.
    """
    ids = {}

    def assign_id(node):
        if node not in ids:
            ids[node] = len(ids)
        node.id = ids[node]

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
