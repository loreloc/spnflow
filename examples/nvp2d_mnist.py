import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from deeprob.flows.models.realnvp import RealNVP2d
from deeprob.torch.transforms import Quantize
from deeprob.torch.routines import train_model, test_model


class UnsupervisedMNIST(datasets.MNIST):
    N_FEATURES = 784
    IMAGE_SIZE = (1, 28, 28)

    def __init__(self, root, train=True, transform=None, download=False):
        super(UnsupervisedMNIST, self).__init__(root, train, transform, download=download)

    def __getitem__(self, index):
        x, y = super(UnsupervisedMNIST, self).__getitem__(index)
        return x


# Set the preprocessing transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    Quantize()
])

# Load the MNIST dataset (unsupervised wrapping) and
# split the dataset in train set and validation set
in_features = UnsupervisedMNIST.IMAGE_SIZE
data_train = UnsupervisedMNIST('datasets', train=True, transform=transform, download=True)
data_test = UnsupervisedMNIST('datasets', train=False, transform=transform, download=True)
n_val = int(0.1 * len(data_train))
n_train = len(data_train) - n_val
data_train, data_val = data.random_split(data_train, [n_train, n_val])

# Instantiate the model
model = RealNVP2d(
    in_features=in_features,
    dequantize=True,
    logit=1e-6,
    n_flows=1,
    n_blocks=2,
    channels=32
)

# Train the model using generative setting (i.e. by maximizing the log-likelihood)
train_model(
    model, data_train, data_val,
    setting='generative',
    batch_size=64,
    optimizer_kwargs={'weight_decay': 5e-5},
    epochs=10, patience=3
)

# Test the model using generative setting
mu_ll, sigma_ll = test_model(model, data_test, 'generative', batch_size=64)
print('Mean LL: {} - Two Stddev LL: {}'.format(mu_ll, sigma_ll))

# Just to make sure to switch to evaluation mode
model.eval()

# Draw some samples
n_samples = 10
samples = model.sample(n_samples ** 2).cpu()
torchvision.utils.save_image(samples / 255.0, 'samples-mnist.png', nrow=n_samples, padding=0)

# Save the model to file
torch.save(model.state_dict(), 'nvp-mnist.pt')
