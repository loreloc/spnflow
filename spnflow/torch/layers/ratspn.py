import torch
import numpy as np


class GaussianLayer(torch.nn.Module):
    """The Gaussian distributions input layer class."""
    def __init__(self, in_features, out_channels, regions, rg_depth, optimize_scale):
        """
        Initialize a Gaussian distributions input layer.

        :param in_features: The number of input features.
        :param out_channels: The number of channels for each base distribution layer.
        :param regions: The regions of the distributions.
        :param rg_depth: The depth of the region graph.
        :param optimize_scale: Whether to optimize scale and location jointly.
        """
        super(GaussianLayer, self).__init__()
        self.in_features = in_features
        self.in_regions = len(regions)
        self.out_channels = out_channels
        self.regions = regions
        self.rg_depth = rg_depth
        self.optimize_scale = optimize_scale

        # Compute the padding and the number of features for each base distribution batch
        self.pad = -self.in_features % (2 ** self.rg_depth)
        self.dimension = (self.in_features + self.pad) // (2 ** self.rg_depth)

        # Append dummy variables to regions orderings and update the pad mask
        mask = self.regions.copy()
        if self.pad > 0:
            pad_mask = np.ones(shape=(self.in_regions, 1, self.dimension), dtype=np.float32)
            for i in range(self.in_regions):
                n_dummy = self.dimension - len(self.regions[i])
                if n_dummy > 0:
                    pad_mask[i, :, -n_dummy:] = 0.0
                    mask[i] = list(mask[i]) + [mask[i][-1]] * n_dummy
            self.register_buffer('pad_mask', torch.tensor(pad_mask))
        self.register_buffer('mask', torch.tensor(mask))

        # Instantiate the location variable
        self.loc = torch.nn.Parameter(
            torch.normal(0.0, 1e-1, size=(self.in_regions, self.out_channels, self.dimension)),
            requires_grad=True
        )

        # Instantiate the scale variable
        if self.optimize_scale:
            self.scale = torch.nn.Parameter(
                torch.normal(0.5, 5e-2, size=(self.in_regions, self.out_channels, self.dimension)),
                requires_grad=True
            )
        else:
            self.scale = torch.nn.Parameter(
                torch.ones(size=(self.in_regions, self.out_channels, self.dimension)),
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
        return torch.sum(x, dim=-1)


class ProductLayer(torch.nn.Module):
    """Product node layer class."""
    def __init__(self, in_regions, in_nodes):
        """
        Initialize the Product layer.

        :param in_regions: The number of input regions.
        :param in_nodes: The number of input nodes per region.
        """
        super(ProductLayer, self).__init__()
        self.in_regions = in_regions
        self.in_nodes = in_nodes
        self.out_partitions = in_regions // 2
        self.out_nodes = in_nodes ** 2

        # Initialize the mask used to compute the outer product
        mask = [True, False] * self.out_partitions
        self.register_buffer('mask', torch.tensor(mask))

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Compute the outer product (the "outer sum" in log domain)
        x1 = x[:,  self.mask]                                  # (-1, out_partitions, in_nodes)
        x2 = x[:, ~self.mask]                                  # (-1, out_partitions, in_nodes)
        x1 = torch.unsqueeze(x1, dim=3)                        # (-1, out_partitions, in_nodes, 1)
        x2 = torch.unsqueeze(x2, dim=2)                        # (-1, out_partitions, 1, in_nodes)
        x = x1 + x2                                            # (-1, out_partitions, in_nodes, in_nodes)
        x = x.view(-1, self.out_partitions, self.out_nodes)    # (-1, out_partitions, out_nodes)
        return x


class SumLayer(torch.nn.Module):
    """Sum node layer."""
    def __init__(self, in_partitions, in_nodes, out_nodes, dropout=None):
        """
        Initialize the sum layer.

        :param in_partitions: The number of input partitions.
        :param in_nodes: The number of input nodes per partition.
        :param out_nodes: The number of output nodes per region.
        :param dropout: The input nodes dropout rate (can be None).
        """
        super(SumLayer, self).__init__()
        self.in_partitions = in_partitions
        self.in_nodes = in_nodes
        self.out_regions = in_partitions
        self.out_nodes = out_nodes
        self.dropout = dropout

        # Instantiate the weights
        self.weight = torch.nn.Parameter(
            torch.normal(0.0, 1e-1, size=(self.out_regions, self.out_nodes, self.in_nodes)),
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
            mask = torch.rand(size=(self.out_regions, self.in_nodes))
            x = x + torch.log(torch.floor(self.dropout + mask))

        # Calculate the log likelihood using the "logsumexp" trick
        w = torch.log_softmax(self.weight, dim=2)  # (out_regions, out_nodes, in_nodes)
        x = torch.unsqueeze(x, dim=2)              # (-1, in_partitions, 1, in_nodes) with in_partitions = out_regions
        x = torch.logsumexp(x + w, dim=-1)         # (-1, out_regions, out_nodes)
        return x


class RootLayer(torch.nn.Module):
    """Root sum node layer."""
    def __init__(self, in_partitions, in_nodes, out_classes):
        """
        Initialize the root layer.

        :param in_partitions: The number of input partitions.
        :param in_nodes: The number of input nodes per partition.
        :param out_classes: The number of output nodes.
        """
        super(RootLayer, self).__init__()
        self.in_partitions = in_partitions
        self.in_nodes = in_nodes
        self.out_classes = out_classes

        # Instantiate the weights
        self.weight = torch.nn.Parameter(
            torch.normal(0.0, 1e-1, size=(self.out_classes, self.in_partitions * self.in_nodes)),
            requires_grad=True
        )

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Flatten the input
        x = torch.flatten(x, start_dim=1)

        # Calculate the log likelihood using the "logsumexp" trick
        w = torch.log_softmax(self.weight, dim=1)  # (out_classes, in_partitions * in_nodes)
        x = torch.unsqueeze(x, dim=1)              # (-1, 1, in_partitions * in_nodes)
        x = torch.logsumexp(x + w, dim=-1)         # (-1, out_classes)
        return x
