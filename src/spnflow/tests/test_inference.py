import unittest
from spnflow.structure.node import *
from spnflow.structure.leaf import *
from spnflow.algorithms.inference import *


class TestInference(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d0 = Bernoulli(0, p=0.3)
        self.d1 = Gaussian(1, mean=0.0, stdev=2.0)
        self.d2 = Gaussian(1, mean=2.0, stdev=1.0)
        self.d3 = Gaussian(2, mean=1.0, stdev=1.0)
        self.d4 = Gaussian(2, mean=3.0, stdev=2.0)
        self.spn = Mul([
            self.d0, Sum([0.3, 0.7], [
                Mul([self.d1, self.d3]),
                Mul([self.d2, self.d4])
            ])
        ])
        assign_ids(self.spn)
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

