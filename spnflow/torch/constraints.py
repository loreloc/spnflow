import torch


class ScaleClipper:
    """
    Constraints the scale to be positive.
    """
    def __init__(self, epsilon=1e-5, **kwargs):
        """
        Initialize the constraint.

        :param epsilon: The epsilon minimum threshold.
        :param kwargs: Other arguments.
        """
        assert epsilon > 0.0
        self.epsilon = torch.tensor(epsilon)

    def to(self, device):
        """
        Move the clipper to the specified device.
        Returns a copy of the current object.

        :param device: The target device to which the clipper should be moved.
        """
        self.epsilon = self.epsilon.to(device=device)

    def __call__(self, module):
        """
        Call the constraint.

        :param module: The module.
        """
        # Clip the scale parameter
        if hasattr(module, 'scale'):
            param = module.scale.data
            param.clamp_(self.epsilon)
