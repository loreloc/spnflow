import unittest
import numpy as np
from spnflow.structure.leaf import *
from spnflow.learning.wrappers import *
from spnflow.algorithms.mpe import *
from spnflow.utils.validity import assert_is_valid
from spnflow.utils.statistics import get_statistics


class TestLearning(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_samples = 500
        x_center = 5.0
        y_center = 10.0
        x = np.random.normal(x_center, 1.0, (n_samples, 2))
        y = np.random.normal(y_center, 1.0, (n_samples, 2))
        c = np.r_[np.zeros((n_samples, 1)), np.ones((n_samples, 1))]
        self.train_data = np.c_[np.r_[x, y], c]
        self.query = [[4.0, 6.0, np.nan], [10.0, 11.0, np.nan]]

    def test_learn_structure(self):
        spn = learn_classifier(self.train_data, [Gaussian, Gaussian, Bernoulli])
        assert_is_valid(spn)
        print(get_statistics(spn))
        classes = [result[2] for result in mpe(spn, self.query)]
        assert classes == [0.0, 1.0]


if __name__ == '__main__':
    unittest.main()

