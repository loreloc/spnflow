import unittest
from spnflow.structure.node import *
from spnflow.structure.leaf import *
from spnflow.utils.validity import assert_is_valid
from spnflow.utils.statistics import get_statistics


class TestStructure(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spn = build_spn()

    def test_get_statistics(self):
        print("SPN Statistics:")
        print(get_statistics(self.spn))

    def test_bfs(self):
        print("BFS IDs:")
        bfs(self.spn, lambda n: print(n.id))

    def test_dfs_post_order(self):
        print("DFS-PO IDs:")
        dfs_post_order(self.spn, lambda n: print(n.id))


def build_spn():
    spn = Sum([0.3, 0.7], [
        Mul([
            Bernoulli(0, p=0.3),
            Sum([0.4, 0.6], [
                Mul([
                    Gaussian(1, mean=0.0, stdev=2.0),
                    Gaussian(2, mean=1.0, stdev=1.0)
                ]),
                Mul([
                    Gaussian(1, mean=2.0, stdev=1.0),
                    Gaussian(2, mean=3.0, stdev=2.0)
                ])
            ])
        ]),
        Mul([
            Bernoulli(0, p=0.3),
            Mul([
                Gaussian(1, mean=0.0, stdev=2.0),
                Gaussian(2, mean=1.0, stdev=1.0),
            ])
        ])
    ])
    assign_ids(spn)
    assert_is_valid(spn)
    return spn


if __name__ == '__main__':
    unittest.main()
