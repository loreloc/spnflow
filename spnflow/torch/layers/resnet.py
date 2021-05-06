import torch
from spnflow.torch.layers.utils import WeightNormConv2d


class ResidualBlock(torch.nn.Module):
    """Residual block for ResNets."""
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):
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
            torch.nn.BatchNorm2d(self.out_channels),
            torch.nn.ReLU(),
            WeightNormConv2d(
                self.in_channels, self.out_channels,
                kernel_size=kernel_size, padding=padding, bias=False
            ),
            torch.nn.BatchNorm2d(self.out_channels),
            torch.nn.ReLU(),
            WeightNormConv2d(
                self.out_channels, self.out_channels,
                kernel_size=kernel_size, padding=padding, bias=True
            )
        )

    def forward(self, x):
        """
        Evaluate the residual block.

        :param x: The inputs.
        :return: The outputs of the block.
        """
        return x + self.block(x)


class ResidualNetwork(torch.nn.Module):
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
        self.blocks = torch.nn.ModuleList()
        self.skips = torch.nn.ModuleList()

        # Build the input convolutional layer and the first skip layer
        self.in_conv = WeightNormConv2d(
            self.in_channels, self.mid_channels,
            kernel_size=(3, 3), padding=(1, 1), bias=True
        )
        self.in_skip = WeightNormConv2d(
            self.mid_channels, self.mid_channels,
            kernel_size=(1, 1), padding=(0, 0), bias=True
        )

        # Build the inner residual blocks and the inner skip layers
        for _ in range(self.n_blocks):
            self.blocks.append(ResidualBlock(
                self.mid_channels, self.mid_channels,
                kernel_size=(3, 3), padding=(1, 1)
            ))
            self.skips.append(WeightNormConv2d(
                self.mid_channels, self.mid_channels,
                kernel_size=(1, 1), padding=(0, 0), bias=True
            ))

        # Build the output batch normalization and output convolutional layers
        self.out_norm = torch.nn.BatchNorm2d(self.mid_channels)
        self.out_conv = WeightNormConv2d(
            self.mid_channels, self.out_channels,
            kernel_size=(1, 1), padding=(0, 0), bias=True
        )

    def forward(self, x):
        """
        Evaluate the residual network.

        :param x: The inputs.
        :return: The outputs of the module.
        """
        # Pass through the input layers
        x = self.in_conv(x)
        z = self.in_skip(x)

        # Apply the inner layers
        for block, skip in zip(self.blocks, self.skips):
            x = block(x)
            z += skip(x)

        # Pass through the output layers
        x = torch.relu(self.out_norm(z))
        return self.out_conv(x)
