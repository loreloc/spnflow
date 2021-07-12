import unittest
import numpy as np

from itertools import product
from deeprob.spn.structure.leaf import Bernoulli
from deeprob.spn.learning.wrappers import learn_estimator
from deeprob.spn.algorithms.inference import log_likelihood
from experiments.datasets import load_binary_dataset


class TestInference(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestInference, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)

        data, _, _ = load_binary_dataset('experiments/datasets', 'nltcs', raw=True)
        n_features = data.shape[1]

        cls.spn_mle = learn_estimator(
            data, [Bernoulli] * n_features, learn_leaf='mle', split_cols='gvs',
            random_state=cls.random_state, verbose=False
        )
        cls.spn_clt = learn_estimator(
            data, [Bernoulli] * n_features, learn_leaf='cltree', split_cols='gvs',
            learn_leaf_kwargs={'to_pc': False}, random_state=cls.random_state, verbose=False
        )

        cls.complete_data = np.array([list(i) for i in product([0, 1], repeat=n_features)])
        indices = cls.random_state.choice(np.arange(len(cls.complete_data)), size=1000, replace=True)
        cls.evi_data = cls.complete_data[indices]
        cls.mar_data = np.copy(cls.evi_data).astype(np.float32)
        mar_mask = cls.random_state.choice([False, True], size=cls.mar_data.shape, p=[0.7, 0.3])
        cls.mar_data[mar_mask] = np.nan

    def test_complete_inference_mle(self):
        lls = log_likelihood(self.spn_mle, self.complete_data)
        self.assertAlmostEqual(np.sum(np.exp(lls)).item(), 1.0, places=5)

    def test_complete_inference_clt(self):
        lls = log_likelihood(self.spn_clt, self.complete_data)
        self.assertAlmostEqual(np.sum(np.exp(lls)).item(), 1.0, places=5)

    def test_marginal_inference_mle(self):
        evi_ll = log_likelihood(self.spn_mle, self.evi_data).mean()
        mar_ll = log_likelihood(self.spn_mle, self.mar_data).mean()
        self.assertGreater(mar_ll, evi_ll)

    def test_marginal_inference_clt(self):
        evi_ll = log_likelihood(self.spn_clt, self.evi_data).mean()
        mar_ll = log_likelihood(self.spn_clt, self.mar_data).mean()
        self.assertGreater(mar_ll, evi_ll)


if __name__ == '__main__':
    unittest.main()
