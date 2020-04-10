import numpy as np


class RegionGraph:
    """
    RegionGraph Class.

    A region graph is defined w.r.t. a set of indices of random variable in a SPN.
    A *region* R is defined as a non-empty subset of the indices, and represented as sorted tuples with unique entries.
    A *partition* P of a region R is defined as a collection of non-empty sets, which are non-overlapping, and whose
    union is R. R is also called *parent* region of P.
    Any region C such that C is in partition P is called *child region* of P.

    So, a region is represented as a sorted tuple of integers (unique elements) and a partition is represented as a
    sorted tuple of regions (non-overlapping, not-empty, at least 2).

    A *region graph* is an acyclic, directed, bi-partite graph over regions and partitions. So, any child of a region
    R is a partition of R, and any child of a partition is a child region of the partition. The root of the region graph
    is a sorted tuple composed of all the elements. The leaves of the region graph must also be regions. They are called
    input regions, or leaf regions.

    Given a region graph, we can easily construct a corresponding SPN:
    1) Associate I distributions to each input region.
    2) Associate K sum nodes to each other (non-input) region.
    3) For each partition P in the region graph, take all cross-products (as product nodes) of distributions/sum nodes
    associated with the child regions. Connect these products as children of all sum nodes in the parent region of P.

    In the end, this procedure will always deliver a complete and decomposable SPN.
    """
    def __init__(self, n_features, depth, seed=42):
        """
        Initialize a region graph.

        :param n_features: The number of features.
        :param depth: The maximum depth.
        :param seed: The random state's seed.
        """
        self._items = tuple(range(n_features))
        self._depth = depth
        self._rnd = np.random.RandomState(seed)

    def random_layers(self):
        """
        Generate a region graph randomly.

        :return: A list of layers, alternating between regions and partitions.
        """
        root = [self._items]
        layers = [root]

        for i in range(self._depth):
            regions = []
            partitions = []
            for r in layers[i * 2]:
                mid = len(r) // 2
                if mid == 0:
                    return layers
                permutation = self._rnd.permutation(r).tolist()
                p0 = tuple(permutation[:mid])
                p1 = tuple(permutation[mid:])
                regions.append(p0)
                regions.append(p1)
                partitions.append((p0, p1))
            layers.append(partitions)
            layers.append(regions)

        return layers


if __name__ == '__main__':
    rg = RegionGraph(17, depth=3)
    layers = rg.random_layers()
    for layer in layers:
        print(layer)
