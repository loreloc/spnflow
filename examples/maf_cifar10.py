import numpy as np
import torch
import torchvision

from spnflow.torch.models import MAF
from spnflow.torch.transforms import Dequantize, Flatten, Reshape
from spnflow.torch.utils import torch_train


class UnsupervisedCIFAR10(torchvision.datasets.CIFAR10):
    IMAGE_SIZE = (3, 32, 32)

    def __init__(self, root, train=True, transform=None, download=False):
        super(UnsupervisedCIFAR10, self).__init__(root, train, transform, download=download)

    def __getitem__(self, index):
        x, y = super(UnsupervisedCIFAR10, self).__getitem__(index)
        return x


# Set the preprocessing transformation (random horizontal flip + dequantization + flatten)
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomHorizontalFlip(),
    Dequantize(),
    Flatten()
])

# Set the inverse transform (used for sampling)
inv_transform = Reshape(UnsupervisedCIFAR10.IMAGE_SIZE)

# Load the CIFAR10 dataset (unsupervised wrapping) and
# split the dataset in train set and validation set
data_train = UnsupervisedCIFAR10('datasets', train=True, transform=transform, download=True)
n_val = int(0.1 * len(data_train))
n_train = len(data_train) - n_val
data_train, data_val = torch.utils.data.random_split(data_train, [n_train, n_val])

# Instantiate the model
# WARNING: very high memory usage
in_features = np.prod(UnsupervisedCIFAR10.IMAGE_SIZE)
model = MAF(
    in_features=in_features,
    n_flows=5,
    units=1024,
    sequential=False,  # Use random degrees
    logit=True         # Apply logit transform
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
torchvision.utils.save_image(images, 'samples_cifar10.png', nrow=n_samples, padding=0)

# Save the model to file
torch.save(model.state_dict(), 'maf_cifar10.pt')
