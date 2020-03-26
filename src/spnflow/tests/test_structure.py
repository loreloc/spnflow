import unittest
from spnflow.structure.node import *
from spnflow.structure.leaf import *


class TestStructure(unittest.TestCase):
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

    def test_bfs(self):
        bfs(self.spn, lambda n: print(n.id))

    def test_dfs_post_order(self):
        dfs_post_order(self.spn, lambda n: print(n.id))


if __name__ == '__main__':
    unittest.main()

