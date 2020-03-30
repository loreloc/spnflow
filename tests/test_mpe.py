import unittest
from spnflow.algorithms.mpe import *
from tests.test_structure import build_spn


class TestInference(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spn = build_spn()
        self.q = [[np.nan, 0.0, 1.0], [np.nan, 2.0, 3.0]]

    def test_mpe(self):
        print(mpe(self.spn, self.q))


if __name__ == '__main__':
    unittest.main()

