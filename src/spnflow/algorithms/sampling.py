import numpy as np
from spnflow.algorithms.mpe import eval_top_down
from spnflow.algorithms.inference import log_likelihood


def sample(root, x):
    _, ls = log_likelihood(root, x, return_results=True)
    return eval_top_down(root, x, ls, lambda n, s: n.sample(s))
