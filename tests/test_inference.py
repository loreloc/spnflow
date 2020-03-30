import unittest
import numpy as np
from spnflow.algorithms.inference import *
from tests.test_structure import build_spn


class TestInference(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spn = build_spn()
        self.q0 = [[0, 0.0, 2.0], [1, 3.0, -2.0]]
        self.q1 = [[0, np.nan, np.nan], [1, np.nan, np.nan]]

    def test_likelihood(self):
        print(likelihood(self.spn, self.q0))
        print(likelihood(self.spn, self.q1))

    def test_log_likelihood(self):
        print(log_likelihood(self.spn, self.q0))
        print(log_likelihood(self.spn, self.q1))


if __name__ == '__main__':
    unittest.main()

