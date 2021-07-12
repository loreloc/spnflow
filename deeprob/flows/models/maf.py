import numpy as np

from deeprob.torch.utils import get_activation_class
from deeprob.flows.utils import BatchNormLayer
from deeprob.flows.layers.autoregressive import AutoregressiveLayer
from deeprob.flows.models.abstract import AbstractNormalizingFlow


class MAF(AbstractNormalizingFlow):
    """Masked Autoregressive Flow (MAF) normalizing flow model."""
    def __init__(self,
                 in_features,
                 dequantize=False,
                 logit=None,
                 in_base=None,
                 n_flows=5,
                 depth=1,
                 units=128,
                 batch_norm=True,
                 activation='relu',
                 sequential=True,
                 random_state=None
                 ):
        """
        Initialize a MAF.

        :param in_features: The number of input features.
        :param dequantize: Whether to apply the dequantization transformation.
        :param logit: The logit factor to use. Use None to disable the logit transformation.
        :param in_base: The input base distribution to use. If None, the standard Normal distribution is used.
        :param n_flows: The number of sequential autoregressive layers.
        :param depth: The number of hidden layers of flows conditioners.
        :param units: The number of hidden units per layer of flows conditioners.
        :param batch_norm: Whether to apply batch normalization after each autoregressive layer.
        :param activation: The activation function name to use for the flows conditioners hidden layers.
        :param sequential: If True build masks degrees sequentially, otherwise randomly.
        :param random_state: The random state used to generate the masks degrees. Used only if sequential is False.
        """
        super(MAF, self).__init__(in_features, dequantize=dequantize, logit=logit, in_base=in_base)
        assert n_flows > 0
        assert depth > 0
        assert units > 0
        self.n_flows = n_flows
        self.depth = depth
        self.units = units
        self.batch_norm = batch_norm
        self.activation = get_activation_class(activation)
        self.sequential = sequential

        # Initialize the random state, if not using sequential masks
        if not self.sequential:
            if random_state is None:
                random_state = np.random.RandomState()
            elif type(random_state) == int:
                random_state = np.random.RandomState(random_state)
            elif not isinstance(random_state, np.random.RandomState):
                raise ValueError("The random state must be either None, a seed integer or a Numpy RandomState")
            self.random_state = random_state
        else:
            self.random_state = None

        # Build the autoregressive layers
        reverse = False
        for _ in range(self.n_flows):
            self.layers.append(
                AutoregressiveLayer(
                    self.in_features, self.depth, self.units, self.activation,
                    reverse=reverse, sequential=self.sequential, random_state=self.random_state
                )
            )

            # Append batch normalization after each layer, if specified
            if self.batch_norm:
                self.layers.append(BatchNormLayer(self.in_features))

            # Invert the input ordering
            reverse = not reverse
