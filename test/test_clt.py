import unittest
import numpy as np

from itertools import product
from deeprob.spn.structure.cltree import BinaryCLTree
from deeprob.spn.algorithms.inference import log_likelihood
from experiments.datasets import load_binary_dataset


class TestCLT(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCLT, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)

        data, _, _ = load_binary_dataset('experiments/datasets', 'nltcs', raw=True)
        n_features = data.shape[1]

        cls.cltree = BinaryCLTree(list(range(n_features)))
        cls.cltree.fit(data, None, alpha=0.1, random_state=cls.random_state)
        cls.spn = cls.cltree.to_pc()

        cls.complete_data = np.array([list(i) for i in product([0, 1], repeat=n_features)])
        indices = cls.random_state.choice(np.arange(len(cls.complete_data)), size=1000, replace=True)
        cls.evi_data = cls.complete_data[indices]
        cls.mar_data = np.copy(cls.evi_data).astype(np.float32)
        mar_mask = cls.random_state.choice([False, True], size=cls.mar_data.shape, p=[0.7, 0.3])
        cls.mar_data[mar_mask] = np.nan

    def test_complete_inference_cltree(self):
        lls = self.cltree.log_likelihood(self.complete_data)
        self.assertAlmostEqual(np.sum(np.exp(lls)).item(), 1.0, places=5)

    def test_marginal_inference_cltree(self):
        evi_ll = self.cltree.log_likelihood(self.evi_data).mean()
        mar_ll = self.cltree.log_likelihood(self.mar_data).mean()
        self.assertGreater(mar_ll, evi_ll)

    def test_inference_pc_conversion(self):
        cltree_evi_ll = self.cltree.log_likelihood(self.evi_data).mean()
        spn_evi_ll = log_likelihood(self.spn, self.evi_data).mean()

        cltree_mar_ll = self.cltree.log_likelihood(self.mar_data).mean()
        spn_mar_ll = log_likelihood(self.spn, self.mar_data).mean()

        self.assertAlmostEqual(cltree_evi_ll, spn_evi_ll, places=5)
        self.assertAlmostEqual(cltree_mar_ll, spn_mar_ll, places=5)


if __name__ == '__main__':
    unittest.main()
