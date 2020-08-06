import torch


class ScaleClipper(torch.nn.Module):
    """
    Constraints the scale to be positive.
    """
    def __init__(self, epsilon=1e-5):
        """
        Initialize the constraint.

        :param epsilon: The epsilon minimum threshold.
        """
        super(ScaleClipper, self).__init__()
        self.register_buffer('epsilon', torch.tensor(epsilon))

    def __call__(self, module):
        """
        Call the constraint.

        :param module: The module.
        """
        # Clip the scale parameter
        module.scale.data.clamp_(self.epsilon)
