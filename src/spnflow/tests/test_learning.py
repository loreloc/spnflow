import unittest
import numpy as np
from spnflow.structure.leaf import *
from spnflow.learning.structure import *
from spnflow.utils.validity import assert_is_valid


class TestInference(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_samples = 50
        x_center = 5.0
        y_center = 10.0
        x = np.random.normal(x_center, 1.0, (n_samples, 2))
        y = np.random.normal(y_center, 1.0, (n_samples, 2))
        c = np.r_[np.zeros((n_samples, 1)), np.ones((n_samples, 1))]
        self.train_data = np.c_[np.r_[x, y], c]

    def test_learn_structure(self):
        spn = learn_structure(self.train_data, [Gaussian, Gaussian, Bernoulli])
        assert_is_valid(spn)


if __name__ == '__main__':
    unittest.main()

