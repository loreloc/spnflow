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
            pad_mask = np.full([self.in_regions, 1, self.dimension], True)
            for i in range(self.in_regions):
                n_dummy = self.dimension - len(self.regions[i])
                if n_dummy > 0:
                    pad_mask[i, :, -n_dummy:] = False
                    mask[i] = tuple(mask[i]) + (mask[i][-1],) * n_dummy
            self.register_buffer('pad_mask', torch.tensor(pad_mask))
        self.register_buffer('mask', torch.tensor(mask))

        # Build the flatten inverse mask
        in_features_pad = self.in_features + self.pad
        inv_mask = np.argsort(np.reshape(mask, [-1, in_features_pad]))
        self.register_buffer('inv_mask', torch.tensor(inv_mask))

        # Build the flatten inverted pad mask
        if self.pad > 0:
            inv_pad_mask = np.reshape(pad_mask, [-1, in_features_pad])
            inv_pad_mask = inv_pad_mask.take(inv_mask)
            self.register_buffer('inv_pad_mask', torch.tensor(inv_pad_mask))

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

        # Initialize the marginalization constant
        self.register_buffer('zero', torch.zeros(1))

    def forward(self, x):
        """
        Execute the layer on some inputs.

        :param x: The inputs.
        :return: The log likelihood of each distribution leaf.
        """
        # Gather the inputs and compute the log-likelihoods
        # Also, marginalize random variables (denoted with NaNs)
        x = torch.unsqueeze(x[:, self.mask], dim=2)
        x = self.distribution.log_prob(x)
        x = torch.where(torch.isnan(x), self.zero, x)
        if self.pad > 0:
            x = torch.where(self.pad_mask, x, self.zero)
        return torch.sum(x, dim=-1)

    @torch.no_grad()
    def mpe(self, x, idx_region, idx_offset):
        """
        Evaluate the layer given some inputs for maximum at posteriori estimation.

        :param x: The inputs. Random variables can be marginalized using NaN values.
        :param idx_region: The region indices.
        :param idx_offset: The offset indices.
        :return: The samples having maximum at posteriori estimates on marginalized random variables.
        """
        idx_repetition = idx_region // (2 ** self.rg_depth)
        idx_repetition = idx_repetition[:, 0]
        n_samples = idx_region.size(0)

        # Get the maximum at posteriori estimation of the base distribution
        mode = self.loc

        # Filter the mode samples by the region and offset indices
        samples = torch.zeros([n_samples, self.in_features], device=self.loc.device)
        for i in range(n_samples):
            j = idx_repetition[i]
            z = torch.flatten(mode[idx_region[i], idx_offset[i]])
            z = z[self.inv_mask[j]]

            # Remove the padding, if required
            if self.pad > 0:
                z = z[self.inv_pad_mask[j]]
            samples[i] = z
        return samples

        # Assign the maximum at posteriori estimation to NaN random variables
        samples = torch.where(torch.isnan(x), samples, x)
        return samples

    @torch.no_grad()
    def sample(self, idx_region, idx_offset):
        """
        Samples some values.

        :param idx_region: The region indices.
        :param idx_offset: The offset indices.
        :return: The samples.
        """
        idx_repetition = idx_region // (2 ** self.rg_depth)
        idx_repetition = idx_repetition[:, 0]
        n_samples = idx_region.size(0)

        # Samples from the full base distribution and reorder it
        x = self.distribution.sample([n_samples])

        # Filter the base samples by the region and offset indices
        samples = torch.zeros([n_samples, self.in_features], device=self.loc.device)
        for i in range(n_samples):
            j = idx_repetition[i]
            z = torch.flatten(x[i, idx_region[i], idx_offset[i]])
            z = z[self.inv_mask[j]]

            # Remove the padding, if required
            if self.pad > 0:
                z = z[self.inv_pad_mask[j]]
            samples[i] = z
        return samples


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

    @torch.no_grad()
    def mpe(self, x, idx_partition, idx_offset):
        """
        Evaluate the layer given some inputs for maximum at posteriori estimation.
        This is equivalent to self.sample() because no parameters are involved in product layers.

        :param x: The inputs. This parameter is a placeholder and it's not used.
        :param idx_partition: the partition indices.
        :param idx_offset: The offset indices.
        :return: The corresponding region and offset indices.
        """
        return self.sample(idx_partition, idx_offset)

    @torch.no_grad()
    def sample(self, idx_partition, idx_offset):
        """
        Sample the partition and offset indices.

        :param idx_partition: the partition indices.
        :param idx_offset: The offset indices.
        :return: The corresponding region and offset indices.
        """
        # Split the region indices
        idx1_region = idx_partition * 2 + 0
        idx2_region = idx_partition * 2 + 1
        idx_region = torch.stack([idx1_region, idx2_region], dim=2)
        idx_region = torch.flatten(idx_region, start_dim=1)

        # Compute the region indices
        idx1_offset = idx_offset // self.in_nodes
        idx2_offset = idx_offset % self.in_nodes
        idx_offset = torch.cat([idx1_offset, idx2_offset], dim=1)

        return idx_region, idx_offset


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

    @torch.no_grad()
    def mpe(self, x, idx_region, idx_offset):
        """
        Evaluate the layer given some inputs for maximum at posteriori estimation.

        :param x: The inputs.
        :param idx_region: The region indices.
        :param idx_offset: The offset indices.
        :return: The corresponding partition and offset indices.
        """
        # Gather both the per-region log-likelihoods and per-node weights
        n_samples = idx_region.size(0)
        n_regions = idx_region.size(1)
        w = torch.log_softmax(self.weight, dim=2)
        u = torch.zeros([n_samples, n_regions, self.in_nodes], device=self.weight.device)
        v = torch.zeros([n_samples, n_regions, self.in_nodes], device=self.weight.device)
        for i in range(n_samples):
            u[i] = x[i, idx_region[i]]
            v[i] = w[idx_region[i], idx_offset[i]]

        # Compute the offset indices evaluating the sum nodes as an argmax
        idx_offset = torch.argmax(u + v, dim=2)

        return idx_region, idx_offset

    @torch.no_grad()
    def sample(self, idx_region, idx_offset):
        """
        Sample the partition and offset indices.

        :param idx_region: The region indices.
        :param idx_offset: The offset indices.
        :return: The corresponding partition and offset indices.
        """
        # Compute the offset indices by sampling from a
        # categorical distribution that is parametrized by sum layer's weights
        n_samples = idx_region.size(0)
        n_regions = idx_region.size(1)
        w = torch.log_softmax(self.weight, dim=2)
        v = torch.zeros([n_samples, n_regions, self.in_nodes], device=self.weight.device)
        for i in range(n_samples):
            v[i] = w[idx_region[i], idx_offset[i]]
        dist = torch.distributions.Categorical(logits=v)
        idx_offset = dist.sample()

        return idx_region, idx_offset


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
        # Calculate the log likelihood using the "logsumexp" trick
        x = torch.flatten(x, start_dim=1)          # (-1, in_partitions * in_nodes)
        w = torch.log_softmax(self.weight, dim=1)  # (out_classes, in_partitions * in_nodes)
        x = torch.unsqueeze(x, dim=1)              # (-1, 1, in_partitions * in_nodes)
        x = torch.logsumexp(x + w, dim=-1)         # (-1, out_classes)
        return x

    @torch.no_grad()
    def mpe(self, x, y):
        """
        Evaluate the layer given some inputs for maximum at posteriori estimation.

        :param x: The inputs.
        :param y: The target classes.
        :return: The partition and offset indices.
        """
        # Compute the layer top-down and get the partitions and nodes indices
        x = torch.flatten(x, start_dim=1)
        w = torch.log_softmax(self.weight, dim=1)
        idx = torch.argmax(x + w[y], dim=1)
        idx_partition = torch.unsqueeze(idx // self.in_nodes, dim=1)
        idx_offset = torch.unsqueeze(idx % self.in_nodes, dim=1)

        return idx_partition, idx_offset

    @torch.no_grad()
    def sample(self, n_samples, y):
        """
        Sample the partition and offset indices.

        :param n_samples: The number of samples.
        :param y: The target classes.
        :return: The partition and offset indices.
        """
        # Get the partitions and nodes indices by sampling from a
        # categorical distribution that is parametrized by root layer's weights
        w = torch.log_softmax(self.weight, dim=1)
        dist = torch.distributions.Categorical(logits=w[y])
        idx = dist.sample()
        idx_partition = torch.unsqueeze(idx // self.in_nodes, dim=1)
        idx_offset = torch.unsqueeze(idx % self.in_nodes, dim=1)

        return idx_partition, idx_offset
