import torch
import torchvision

from spnflow.torch.models.flows import RealNVP2d
from spnflow.torch.transforms import Quantize
from spnflow.torch.routines import train


class UnsupervisedCIFAR10(torchvision.datasets.CIFAR10):
    N_FEATURES = 3072
    IMAGE_SIZE = (3, 32, 32)

    def __init__(self, root, train=True, transform=None, download=False):
        super(UnsupervisedCIFAR10, self).__init__(root, train, transform, download=download)

    def __getitem__(self, index):
        x, y = super(UnsupervisedCIFAR10, self).__getitem__(index)
        return x


# Set the preprocessing transformation
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomHorizontalFlip(),
    Quantize()
])

# Load the CIFAR-10 dataset (unsupervised wrapping) and
# split the dataset in train set and validation set
in_features = UnsupervisedCIFAR10.IMAGE_SIZE
data_train = UnsupervisedCIFAR10('datasets', train=True, transform=transform, download=True)
n_val = int(0.1 * len(data_train))
n_train = len(data_train) - n_val
data_train, data_val = torch.utils.data.random_split(data_train, [n_train, n_val])

# Instantiate the model
model = RealNVP2d(
    in_features=in_features,
    dequantize=True,
    logit=0.05,
    n_flows=1,
    n_blocks=4,
    channels=64
)

# Train the model using generative setting (i.e. by maximizing the log-likelihood)
train(
    model, data_train, data_val,
    setting='generative',
    batch_size=64,
    weight_decay=5e-5,
    epochs=10,
    patience=3
)

# Just to make sure to switch to evaluation mode
model.eval()

# Draw some samples
n_samples = 10
samples = model.sample(n_samples ** 2).cpu()
torchvision.utils.save_image(samples / 255.0, 'samples-cifar10.png', nrow=n_samples, padding=0)

# Save the model to file
torch.save(model.state_dict(), 'nvp-cifar10.pt')
