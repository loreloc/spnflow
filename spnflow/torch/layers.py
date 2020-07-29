import numpy as np
import torch


class GaussianLayer(torch.nn.Module):
    """The Gaussian distributions input layer class."""
    def __init__(self, n_features, depth, regions, n_batch, optimize_scale):
        """
        Initialize a Gaussian distributions input layer.

        :param n_features: The number of features.
        :param depth: The depth of the RAT-SPN.
        :param regions: The regions of the distributions.
        :param n_batch: The number of distributions.
        :param optimize_scale: Whatever to train scale and mean jointly.
        """
        super(GaussianLayer, self).__init__()
        self.n_features = n_features
        self.depth = depth
        self.regions = regions
        self.n_batch = n_batch
        self.optimize_scale = optimize_scale
        self.n_regions = len(self.regions)

        # Compute the padding
        self.pad = -self.n_features % (2 ** self.depth)
        self.dim_gauss = (self.n_features + self.pad) // (2 ** self.depth)

        # Append dummy variables to regions orderings and update the pad mask
        mask = self.regions.copy()
        if self.pad > 0:
            pad_mask = np.ones(shape=(self.n_regions, 1, self.dim_gauss), dtype=np.float32)
            for i in range(self.n_regions):
                n_dummy = self.dim_gauss - len(self.regions[i])
                if n_dummy > 0:
                    pad_mask[i, :, -n_dummy:] = 0.0
                    mask[i] = list(mask[i]) + [list(mask[i])[-1]] * n_dummy
            self.pad_mask = torch.nn.Parameter(torch.tensor(pad_mask), requires_grad=False)
        self.mask = torch.nn.Parameter(torch.tensor(mask), requires_grad=False)

        # Instantiate the location variable
        self.loc = torch.nn.Parameter(
            torch.normal(0.0, 1e-1, size=(self.n_regions, self.n_batch, self.dim_gauss)),
            requires_grad=True
        )

        # Instantiate the scale variable
        if self.optimize_scale:
            self.scale = torch.nn.Parameter(
                torch.normal(0.5, 5e-2, size=(self.n_regions, self.n_batch, self.dim_gauss)),
                requires_grad=True
            )
        else:
            self.scale = torch.nn.Parameter(
                torch.ones(size=(self.n_regions, self.n_batch, self.dim_gauss)),
                requires_grad=False
            )

        # Instantiate the multi-batch normal distribution
        self.distribution = torch.distributions.Normal(self.loc, self.scale)

    def forward(self, x):
        """
        Execute the layer on some inputs.

        :param x: The inputs.
        :return: The log likelihood of each distribution leaf.
        """
        # Gather the inputs and compute the log-likelihoods
        x = torch.unsqueeze(x[:, self.mask], dim=2)
        x = self.distribution.log_prob(x)
        if self.pad > 0:
            x = x * self.pad_mask
        x = x.sum(dim=-1)
        return x


class ProductLayer(torch.nn.Module):
    """Product node layer class."""
    def __init__(self, n_partitions, n_nodes):
        """
        Initialize the Product layer.

        :param n_partitions: The number of partitions.
        :param n_nodes: The number of child nodes for each region.
        """
        super(ProductLayer, self).__init__()
        self.n_partitions = n_partitions
        self.n_nodes = n_nodes

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Compute the outer product (the "outer sum" in log domain)
        x = torch.reshape(x, (-1, self.n_partitions, 2, self.n_nodes))  # (n, p, 2, s)
        x0 = torch.unsqueeze(x[:, :, 0], dim=3)  # (n, p, s, 1)
        x1 = torch.unsqueeze(x[:, :, 1], dim=2)  # (n, p, 1, s)
        x = x0 + x1  # (n, p, s, s)
        x = torch.reshape(x, [-1, self.n_partitions, self.n_nodes ** 2])  # (n, p, s * s)
        return x


class SumLayer(torch.nn.Module):
    """Sum node layer."""
    def __init__(self, n_regions, n_nodes, n_sum, dropout=None):
        """
        Initialize the sum layer.

        :param n_regions: The number of regions.
        :param n_nodes: The number of child nodes for each region.
        :param n_sum: The number of sum node per region.
        :param dropout: The input nodes dropout rate (can be None).
        """
        super(SumLayer, self).__init__()
        self.n_regions = n_regions
        self.n_nodes = n_nodes
        self.n_sum = n_sum
        self.dropout = dropout

        # Instantiate the weights
        self.weight = torch.nn.Parameter(
            torch.normal(0.0, 5e-1, size=(self.n_regions, self.n_sum, self.n_nodes)),
            requires_grad=True
        )

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Apply the dropout, if specified
        if self.training and self.dropout is not None:
            mask = torch.rand(size=(self.n_regions, self.n_nodes))
            x = x + torch.log(torch.floor(self.dropout + mask))

        # Calculate the log likelihood using the "logsumexp" trick
        x = torch.unsqueeze(x, dim=2)
        w = torch.log_softmax(self.weight, dim=2)  # (p, s)
        x = torch.logsumexp(x + w, dim=-1)  # (n, p, s)
        return x


class RootLayer(torch.nn.Module):
    """Root sum node layer."""
    def __init__(self, n_sum, n_nodes):
        """
        Initialize the root layer.

        :param n_sum: The number of sum nodes.
        :param n_nodes: The number of input nodes.
        """
        super(RootLayer, self).__init__()
        self.n_sum = n_sum
        self.n_nodes = n_nodes

        # Instantiate the weights
        self.kernel = torch.nn.Parameter(
            torch.normal(0.0, 5e-1, size=(self.n_sum, self.n_nodes)),
            requires_grad=True
        )

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Calculate the log likelihood using the "logsumexp" trick
        x = torch.unsqueeze(x, dim=1)
        w = torch.log_softmax(self.kernel, dim=1)
        x = torch.logsumexp(x + w, dim=-1)
        return x
