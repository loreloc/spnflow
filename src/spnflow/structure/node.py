import numpy as np
from collections import deque


class Node:
    def __init__(self, scope, children):
        self.id = None
        self.scope = scope
        self.children = children

    def likelihood(self, x):
        pass

    def log_likelihood(self, x):
        pass


class Sum(Node):
    def __init__(self, weights, children):
        scope = children[0].scope
        assert len(weights) > 0, "Weights list cannot be empty"
        assert len(children) > 0, "Children list cannot be empty"
        assert len(weights) == len(children), "Each child must be associated to a weight"
        assert np.isclose(sum(weights), 1.0), "The sum of weights must be 1.0"
        assert all(set(c.scope) == set(scope) for c in children), "The scopes of the children must be equal"
        super().__init__(scope, children)
        self.weights = weights

    def likelihood(self, x):
        return np.dot(x, self.weights).reshape(-1, 1)

    def log_likelihood(self, x):
        z = np.exp(x)
        z[np.isclose(z, 0.0)] = np.finfo(float).eps
        return np.log(np.dot(z, self.weights)).reshape(-1, 1)


class Mul(Node):
    def __init__(self, children):
        scope = sum([c.scope for c in children], [])
        assert len(children) > 0, "Children list cannot be empty"
        assert set(scope) == set.union(*[set(c.scope) for c in children]), "The scopes of the children must be disjoint"
        super().__init__(scope, children)

    def likelihood(self, x):
        return np.prod(x, axis=1).reshape(-1, 1)

    def log_likelihood(self, x):
        z = np.copy(x)
        z[np.isinf(z)] = np.finfo(float).min
        return np.sum(z, axis=1).reshape(-1, 1)


def assign_ids(root):
    ids = {}

    def assign_id(node):
        if node not in ids:
            ids[node] = len(ids)
        node.id = ids[node]

    bfs(root, assign_id)
    return root


def bfs(root, func):
    seen, queue = {root}, deque([root])
    while queue:
        node = queue.popleft()
        func(node)
        for c in node.children:
            if c not in seen:
                seen.add(c)
                queue.append(c)


def dfs_post_order(root, func):
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
