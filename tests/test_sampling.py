import unittest
import numpy as np
from spnflow.algorithms.sampling import *
from tests.test_structure import build_spn


class TestInference(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spn = build_spn()
        self.q0 = [[np.nan, 0.0, 1.0], [np.nan, 2.0, 3.0]]
        self.q1 = [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]

    def test_sample(self):
        print(sample(self.spn, self.q0))
        print(sample(self.spn, self.q1))


if __name__ == '__main__':
    unittest.main()

