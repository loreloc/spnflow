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
    def __init__(self, items, seed=42):
        """
        Initialize a region graph.

        :param items: The items of the root region.
        :param seed: The random state's seed.
        """
        self._items = tuple(sorted(items))
        self._rnd = np.random.RandomState(seed)
        self._regions = {self._items}
        self._children = dict()
        self._partitions = set()
        self._layers = list()

    def items(self):
        """
        Get the items of the region graph.

        :return: The items.
        """
        return self._items

    def regions(self):
        """
        Get the regions of the region graph.

        :return: The regions.
        """
        return self._regions

    def children(self):
        """
        Get the children dictionary of the region graph.

        :return: The children dictionary.
        """
        return self._children

    def partitions(self):
        """
        Get the partitions of the region graph.

        :return: The partitions.
        """
        return self._partitions

    def layers(self):
        """
        Get the layers of the region graph. The returned value is an empty list if *make_layers* is not called.

        :return: The layers.
        """
        return self._layers

    def leaves(self):
        """
        Get the leaves of the region graph.

        :return: The leaves.
        """
        return [x for x in self._regions if x not in self._children]

    def random_split(self, depth):
        """
        Split the ragion graph randomly given a certain depth.

        :param depth: The maximum depth.
        """
        assert depth > 0, "The depth must be greater than zero"

        # Start with the root region
        tasks = list()
        tasks.append((self._items, depth))

        while tasks:
            # Permute and split in half the region
            (items, depth) = tasks.pop()
            items = self._rnd.permutation(items).tolist()
            mid = len(items) // 2
            r1 = tuple(sorted(items[:mid]))
            r2 = tuple(sorted(items[mid:]))

            # Create a partition from the region split
            partition = tuple(sorted((r1, r2)))
            self._regions.add(r1)
            self._regions.add(r2)

            # Update the partitions list and the children dictionary
            parent = tuple(sorted(x for child in partition for x in child))
            if partition not in self._partitions:
                self._partitions.add(partition)
                children = self._children.get(parent, [])
                self._children[parent] = children + [partition]

            # Recursively run the random split over the sub-regions
            if depth > 1:
                if len(r1) > 1:
                    tasks.append((r1, depth - 1))
                if len(r2) > 1:
                    tasks.append((r2, depth - 1))

    def make_layers(self):
        """
        Make a layered structure, represented as a list of lists.

        The layered representation is greedily constructed, in order to contain as few layers as possible.
        Crucially it repstects some topological order of the region graph (a directed graph is acyclic if and only if
        there exists a topological order of its node), i.e. if k >= 1, it is guaranteed that regions (or partitions) in
        layer k cannot be children of partitions (or regions) in layer l.

        _layers[0] will contain leaf regions.
        For k > 0:
            _layers[k] will contain partitions, if k is odd
            _layers[k] will contain regions, if k is even
        """
        seen_regions = set()
        seen_partitions = set()

        # Start with the first layer composed by the leaves of the region graph
        leaves = sorted(self.leaves())
        self._layers = [leaves]
        seen_regions.update(leaves)

        # While there are regions or partitions to process ...
        while len(seen_regions) != len(self._regions) or len(seen_partitions) != len(self.partitions()):
            # The next partition layer contains all partitions which have not been visited and
            # all its child regions have been visited
            next_layer = [p for p in self._partitions
                          if p not in seen_partitions and all([r in seen_regions for r in p])]
            self._layers.append(next_layer)
            seen_partitions.update(next_layer)

            # The next region layer contains all regions which have not been visited and
            # all its child partitions have been visited.
            next_layer = [r for r in self._regions
                          if r not in seen_regions and all([p in seen_partitions for p in self._children[r]])]
            next_layer = sorted(next_layer)
            self._layers.append(next_layer)
            seen_regions.update(next_layer)


if __name__ == '__main__':
    rg = RegionGraph([0, 1, 2, 3, 4, 5, 6, 7, 8])

    for i in range(3):
        rg.random_split(2)
    rg.make_layers()
    layers = rg.layers()
    for layer in reversed(layers):
        print(layer)
