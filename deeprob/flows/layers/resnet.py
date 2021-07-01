import torch.nn as nn

from deeprob.torch.utils import WeightNormConv2d


class ResidualBlock(nn.Module):
    """The residual basic block as in ResNet."""
    def __init__(self, n_channels):
        """
        Build a residual block.

        :param n_channels: The number of channels.
        """
        super(ResidualBlock, self).__init__()
        self.n_channels = n_channels

        # Build the residual block
        self.block = nn.Sequential(
            nn.BatchNorm2d(self.n_channels),
            nn.ReLU(inplace=True),
            WeightNormConv2d(
                self.n_channels, self.n_channels,
                kernel_size=(3, 3), padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(self.n_channels),
            nn.ReLU(inplace=True),
            WeightNormConv2d(
                self.n_channels, self.n_channels,
                kernel_size=(3, 3), padding=(1, 1), bias=False
            )
        )

    def forward(self, x):
        """
        Evaluate the residual block.

        :param x: The inputs tensor.
        :return: The outputs.
        """
        return x + self.block(x)


class ResidualNetwork(nn.Module):
    """Residual network (aka ResNet) with skip connections."""
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
        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()

        # Build the input convolutional layer and input skip layer
        self.in_conv = WeightNormConv2d(
            self.in_channels, self.mid_channels,
            kernel_size=(3, 3), padding=(1, 1), bias=False
        )
        self.in_skip = WeightNormConv2d(
            self.mid_channels, self.mid_channels,
            kernel_size=(1, 1), padding=(0, 0), bias=True
        )

        # Build the list of residual blocks
        for _ in range(self.n_blocks):
            self.blocks.append(ResidualBlock(self.mid_channels))
            self.skips.append(WeightNormConv2d(
                self.mid_channels, self.mid_channels,
                kernel_size=(1, 1), padding=(0, 0), bias=True
            ))

        # Build the output network
        self.out_network = nn.Sequential(
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
            WeightNormConv2d(
                self.mid_channels, self.out_channels,
                kernel_size=(1, 1), padding=(0, 0), bias=True
            )
        )

    def forward(self, x):
        """
        Evaluate the residual network.

        :param x: The inputs.
        :return: The outputs of the module.
        """
        # Pass through the input convolutional layer
        x = self.in_conv(x)
        z = self.in_skip(x)

        # Pass through the residual blocks
        for block, skip in zip(self.blocks, self.skips):
            x = block(x)
            z += skip(x)

        # Pass through the output network
        x = self.out_network(z)
        return x
