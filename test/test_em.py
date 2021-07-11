import unittest
import numpy as np

from sklearn.datasets import make_blobs
from deeprob.spn.structure.node import Sum, Mul, assign_ids
from deeprob.spn.structure.leaf import Bernoulli, Gaussian
from deeprob.spn.learning.wrappers import learn_estimator
from deeprob.spn.algorithms.inference import log_likelihood
from deeprob.spn.learning.em import expectation_maximization
from experiments.datasets import load_binary_dataset


class TestEM(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEM, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)

        cls.binary_data, _, _ = load_binary_dataset('experiments/datasets', 'nltcs', raw=True)
        blobs_data, _ = make_blobs(n_samples=500, n_features=2, centers=2, random_state=cls.random_state)
        cls.blobs_data = (blobs_data - np.mean(blobs_data, axis=0)) / np.std(blobs_data, axis=0)

        n_features = cls.binary_data.shape[1]
        cls.binary_spn = learn_estimator(
            cls.binary_data, [Bernoulli] * n_features,
            learn_leaf='mle', split_cols='gvs', verbose=False
        )

        cls.blobs_spn = Sum(None, [
            Mul([Gaussian(0), Gaussian(1)]),
            Mul([Gaussian(0), Gaussian(1)])
        ])
        assign_ids(cls.blobs_spn)

    def test_binary_em(self):
        before_lls = log_likelihood(self.binary_spn, self.binary_data)
        expectation_maximization(
            self.binary_spn, self.binary_data, num_iter=25, batch_perc=0.2,
            random_init=False, verbose=False
        )
        after_lls = log_likelihood(self.binary_spn, self.binary_data)
        self.assertGreater(np.mean(after_lls), np.mean(before_lls))

    def test_gaussian_em(self):
        expectation_maximization(
            self.blobs_spn, self.blobs_data, num_iter=50, batch_perc=0.2,
            random_init=True, random_state=self.random_state, verbose=False
        )
        lls = log_likelihood(self.blobs_spn, self.blobs_data)
        self.assertAlmostEqual(np.mean(lls).item(), -0.89, places=3)


if __name__ == '__main__':
    unittest.main()
