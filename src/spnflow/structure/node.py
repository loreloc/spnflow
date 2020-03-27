import numpy as np
from collections import deque


class Node:
    def __init__(self, children, scope):
        self.id = None
        self.children = children
        self.scope = scope

    def likelihood(self, x):
        pass

    def log_likelihood(self, x):
        pass


class Sum(Node):
    def __init__(self, weights, children=[], scope=[]):
        if len(scope) == 0 and len(children) > 0:
            scope = children[0].scope
        super().__init__(children, scope)
        self.weights = weights

    def likelihood(self, x):
        return np.dot(x, self.weights).reshape(-1, 1)

    def log_likelihood(self, x):
        z = np.exp(x)
        z[np.isclose(z, 0.0)] = np.finfo(float).eps
        return np.log(np.dot(z, self.weights)).reshape(-1, 1)


class Mul(Node):
    def __init__(self, children=[], scope=[]):
        if len(scope) == 0 and len(children) > 0:
            scope = sum([c.scope for c in children], [])
        super().__init__(children, scope)

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
