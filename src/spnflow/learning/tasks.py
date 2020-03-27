from enum import Enum
from collections import deque


class OperationKind(Enum):
    CREATE_LEAF = 1,
    SPLIT_ROWS = 2,
    SPLIT_COLS = 3,
    NAIVE_FACTORIZE = 4,
    ROOT_SPLIT_ROWS = 5,
    ROOT_SPLIT_COLS = 6


class TaskQueue:
    def __init__(self, min_rows_slice, min_cols_slice):
        self.tasks = deque()
        self.min_rows_slice = min_rows_slice
        self.min_cols_slice = min_cols_slice

    def __len__(self):
        return len(self.tasks)

    def push(self, parent, op, local_data, scope):
        self.tasks.append((parent, op, local_data, scope))

    def pop(self):
        task = (parent, op, local_data, scope) = self.tasks.popleft()
        n_rows, n_cols = local_data.shape
        if n_rows >= self.min_rows_slice and n_cols >= self.min_cols_slice:
            return task
        return parent, OperationKind.NAIVE_FACTORIZE, local_data, scope
