import torch
from torch.utils.checkpoint import checkpoint
from spnflow.torch.layers.utils import WeightNormConv2d


class DenseLayer(torch.nn.Module):
    """The dense layer as in DenseNet."""
    def __init__(self, in_channels, out_channels, use_checkpoint=False):
        """
        Initialize a dense layer.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param use_checkpoint: Whether to use a checkpoint in order to reduce memory usage
                               (by increasing training time caused by re-computations).
        """
        super(DenseLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = 4 * self.out_channels  # Use 4 * out_channels as number of mid features channels
        self.use_checkpoint = use_checkpoint

        # Build the bottleneck network
        self.bottleneck_network = torch.nn.Sequential(
            torch.nn.BatchNorm2d(self.in_channels),
            torch.nn.ReLU(inplace=True),
            WeightNormConv2d(
                self.in_channels, self.mid_channels,
                kernel_size=(1, 1), padding=(0, 0), bias=False
            )
        )

        # Build the main dense layer
        self.network = torch.nn.Sequential(
            torch.nn.BatchNorm2d(self.mid_channels),
            torch.nn.ReLU(inplace=True),
            WeightNormConv2d(
                self.mid_channels, self.out_channels,
                kernel_size=(3, 3), padding=(1, 1), bias=False
            )
        )

    def bottleneck(self, inputs):
        """
        Pass through the bottleneck layer.

        :param inputs: A list of previous feature maps.
        :return: The outputs of the bottleneck.
        """
        x = torch.cat(inputs, dim=1)
        return self.bottleneck_network(x)

    def checkpoint_bottleneck(self, inputs):
        """
        Pass through the bottleneck layer (by using a checkpoint).

        :param inputs: A list of previous feature maps.
        :return: The outputs of the bottleneck.
        """
        def closure(*inputs):
            return self.bottleneck(inputs)
        return checkpoint(closure, *inputs)

    def forward(self, inputs):
        """
        Evaluate the dense layer.

        :param inputs: A list of previous feature maps.
        :return: The outputs of the layer.
        """
        # Pass through the bottleneck
        if self.use_checkpoint and any(map(lambda t: t.requires_grad, inputs)):
            x = self.checkpoint_bottleneck(inputs)
        else:
            x = self.bottleneck(inputs)

        # Pass through the main dense layer
        x = self.network(x)
        return x


class DenseBlock(torch.nn.Module):
    """The dense block as in DenseNet."""
    def __init__(self, n_layers, in_channels, out_channels, use_checkpoint):
        """
        Initialize a dense block.

        :param n_layers: The number of dense layers.
        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param use_checkpoint: Whether to use a checkpoint in order to reduce memory usage
                              (by increasing training time caused by re-computations).
        """
        super(DenseBlock, self).__init__()
        self.layers = torch.nn.ModuleList()

        # Build the dense layers
        for i in range(n_layers):
            self.layers.append(DenseLayer(
                in_channels + i * out_channels, out_channels, use_checkpoint
            ))

    def forward(self, x):
        """
        Evaluate the dense block.

        :param x: The inputs.
        :return: The outputs of the layer.
        """
        outputs = [x]
        for layer in self.layers:
            x = layer(outputs)
            outputs.append(x)
        return torch.cat(outputs, dim=1)


class Transition(torch.nn.Module):
    """The transition layer as in DenseNet."""
    def __init__(self, in_channels, out_channels, bias=True):
        """
        Initialize a transition layer.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param bias: Whether to use bias in the last convolutional layer.
        """
        super(Transition, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Build the transition layer
        self.network = torch.nn.Sequential(
            torch.nn.BatchNorm2d(self.in_channels),
            torch.nn.ReLU(inplace=True),
            WeightNormConv2d(
                self.in_channels, self.out_channels,
                kernel_size=(1, 1), padding=(0, 0), bias=bias
            )
        )

    def forward(self, x):
        """
        Evaluate the layer.

        :param x: The inputs.
        :return: The outputs of the layer.
        """
        return self.network(x)


class DenseNetwork(torch.nn.Module):
    """Dense network (aka DenseNet) with only one dense block."""
    def __init__(self, in_channels, mid_channels, out_channels, n_blocks, use_checkpoint=False):
        """
        Initialize a dense network.

        :param in_channels: The number of input channels.
        :param mid_channels: The number of mid channels.
        :param out_channels: The number of output channels.
        :param n_blocks: The number of dense blocks.
        :param use_checkpoint: Whether to use a checkpoint in order to reduce memory usage
                              (by increasing training time caused by re-computations).
        """
        super(DenseNetwork, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.blocks = torch.nn.ModuleList()
        self.n_layers = 4  # Use four dense layer for each dense block

        # Build the input convolutional layer
        self.in_conv = WeightNormConv2d(
            self.in_channels, self.mid_channels,
            kernel_size=(3, 3), padding=(1, 1), bias=False
        )

        # Build the list of dense blocks and transition layers
        for i in range(self.n_blocks):
            self.blocks.append(DenseBlock(
                self.n_layers, self.mid_channels, self.mid_channels, use_checkpoint
            )),
            if i == self.n_blocks - 1:
                self.blocks.append(Transition(
                    (self.n_layers + 1) * self.mid_channels, self.out_channels, bias=True
                ))
            else:
                self.blocks.append(Transition(
                    (self.n_layers + 1) * self.mid_channels, self.mid_channels, bias=False
                ))

    def forward(self, x):
        """
        Evaluate the dense network.

        :param x: The inputs.
        :return: The outputs of the model.
        """
        # Pass through the input convolutional layer
        x = self.in_conv(x)

        # Pass through the dense blocks
        for block in self.blocks:
            x = block(x)
        return x
