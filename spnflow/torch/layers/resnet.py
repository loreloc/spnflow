import torch


class ResidualBlock(torch.nn.Module):
    """Residual block for ResNets."""
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        """
        Initialize a residual block as in ResNets.

        :param in_channels: The number of channels of the inputs.
        :param out_channels: The number of channels of the outputs.
        :param kernel_size: The kernel size.
        :param padding: The padding size to use.
        """
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, padding=padding, bias=False),
            torch.nn.BatchNorm2d(self.in_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            torch.nn.BatchNorm2d(self.out_channels),
        )
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        """
        Evaluate the residual block.

        :param x: The inputs.
        :return: The outputs of the block.
        """
        return self.activation(x + self.block(x))


class ResidualNetwork(torch.nn.Module):
    """Residual network (aka ResNet)."""
    def __init__(self, in_channels, mid_channels, out_channels, n_blocks):
        """
        Initialize a residual network.

        :param in_channels: The number of input channels.
        :param mid_channels: The number of mid channels.
        :param out_channels: The number of output channels.
        :param n_blocks: The number of residual blocks.
        """
        super(ResidualNetwork, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.layers = torch.nn.ModuleList()

        # Build the input layers
        self.layers.extend([
            torch.nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(self.mid_channels),
            torch.nn.ReLU()
        ])

        # Build the inner residual blocks
        for _ in range(self.n_blocks):
            self.layers.append(
                ResidualBlock(self.mid_channels, self.mid_channels, kernel_size=3, padding=1)
            )

        # Build the output layer
        self.layers.append(
            torch.nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=1, padding=0)
        )

    def forward(self, x):
        """
        Evaluate the residual network.

        :param x: The inputs.
        :return: The outputs of the module.
        """
        for layer in self.layers:
            x = layer(x)
        return x
