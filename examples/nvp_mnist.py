import numpy as np
import torch
import torchvision

from spnflow.torch.models.flows import RealNVP1d
from spnflow.torch.transforms import Dequantize, Flatten, Reshape
from spnflow.torch.routines import torch_train


class UnsupervisedMNIST(torchvision.datasets.MNIST):
    IMAGE_SIZE = (1, 28, 28)

    def __init__(self, root, train=True, transform=None, download=False):
        super(UnsupervisedMNIST, self).__init__(root, train, transform, download=download)

    def __getitem__(self, index):
        x, y = super(UnsupervisedMNIST, self).__getitem__(index)
        return x


# Set the preprocessing transformation (dequantization + flatten)
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    Dequantize(),
    Flatten()
])

# Set the inverse transform (used for sampling)
inv_transform = Reshape(UnsupervisedMNIST.IMAGE_SIZE)

# Load the MNIST dataset (unsupervised wrapping) and
# split the dataset in train set and validation set
data_train = UnsupervisedMNIST('datasets', train=True, transform=transform, download=True)
n_val = int(0.1 * len(data_train))
n_train = len(data_train) - n_val
data_train, data_val = torch.utils.data.random_split(data_train, [n_train, n_val])

# Instantiate the model
# WARNING: very high memory usage
in_features = np.prod(UnsupervisedMNIST.IMAGE_SIZE).item()
model = RealNVP1d(
    in_features=in_features,
    n_flows=5,
    units=1024,
    logit=True  # Apply logit transform
)

# Train the model using generative setting (i.e. by maximizing the log-likelihood)
torch_train(
    model, data_train, data_val,
    setting='generative',
    weight_decay=1e-6,
    epochs=100,
    patience=10
)

# Draw some samples
n_samples = 10
samples = model.sample(n_samples ** 2).cpu()
images = torch.stack([inv_transform(x) for x in samples], dim=0)
torchvision.utils.save_image(images, 'samples_mnist.png', nrow=n_samples, padding=0)

# Save the model to file
torch.save(model.state_dict(), 'nvp_mnist.pt')
