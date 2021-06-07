import unittest
import numpy as np

from itertools import product
from deeprob.structure.leaf import Bernoulli
from deeprob.learning.wrappers import learn_estimator
from deeprob.algorithms.inference import log_likelihood
from experiments.datasets import load_binary_dataset


class TestInference(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestInference, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        data, _, _ = load_binary_dataset('experiments/datasets', 'nltcs', raw=True)
        n_features = data.shape[1]
        cls.spn_mle = learn_estimator(
            data, [Bernoulli] * n_features,
            learn_leaf='mle', split_cols='gvs', verbose=False
        )
        cls.spn_clt = learn_estimator(
            data, [Bernoulli] * n_features,
            learn_leaf='cltree', split_cols='gvs', learn_leaf_kwargs={'to_pc': True}, verbose=False
        )
        cls.spn_clt_plain = learn_estimator(
            data, [Bernoulli] * n_features,
            learn_leaf='cltree', split_cols='gvs', learn_leaf_kwargs={'to_pc': False}, verbose=False
        )
        cls.complete_data = np.array([list(i) for i in product([0, 1], repeat=n_features)])

    def test_complete_inference_mle(self):
        ll = log_likelihood(self.spn_mle, self.complete_data)
        self.assertAlmostEqual(np.sum(np.exp(ll)).item(), 1.0, places=6)

    def test_complete_inference_clt(self):
        ll = log_likelihood(self.spn_clt, self.complete_data)
        self.assertAlmostEqual(np.sum(np.exp(ll)).item(), 1.0, places=6)

    def test_complete_inference_clt_plain(self):
        ll = log_likelihood(self.spn_clt_plain, self.complete_data)
        self.assertAlmostEqual(np.sum(np.exp(ll)).item(), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
