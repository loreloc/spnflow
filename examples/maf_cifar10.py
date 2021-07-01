import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from deeprob.flows.models.maf import MAF
from deeprob.torch.transforms import Quantize, Flatten, Reshape
from deeprob.torch.routines import train_model, test_model


class UnsupervisedCIFAR10(datasets.CIFAR10):
    N_FEATURES = 3072
    IMAGE_SIZE = (3, 32, 32)

    def __init__(self, root, train=True, transform=None, download=False):
        super(UnsupervisedCIFAR10, self).__init__(root, train, transform, download=download)

    def __getitem__(self, index):
        x, y = super(UnsupervisedCIFAR10, self).__getitem__(index)
        return x


# Set the preprocessing transformation (quantization + random horizontal flip + flatten)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    Quantize(),
    Flatten()
])

# Set the inverse transform (used for sampling)
inv_transform = Reshape(UnsupervisedCIFAR10.IMAGE_SIZE)

# Load the CIFAR10 dataset (unsupervised wrapping) and
# split the dataset in train set and validation set
in_features = UnsupervisedCIFAR10.N_FEATURES
data_train = UnsupervisedCIFAR10('datasets', train=True, transform=transform, download=True)
data_test = UnsupervisedCIFAR10('datasets', train=False, transform=transform, download=True)
n_val = int(0.1 * len(data_train))
n_train = len(data_train) - n_val
data_train, data_val = data.random_split(data_train, [n_train, n_val])

# Instantiate the model
model = MAF(
    in_features=in_features,
    dequantize=True,
    logit=0.05,
    n_flows=3,
    depth=1,
    units=1024,
    sequential=False
)

# Train the model using generative setting (i.e. by maximizing the log-likelihood)
train_model(
    model, data_train, data_val,
    setting='generative',
    batch_size=64,
    lr=1e-4,
    optimizer_kwargs={'weight_decay': 5e-5},
    epochs=20, patience=3
)

# Test the model using generative setting
mu_ll, sigma_ll = test_model(model, data_test, 'generative', batch_size=64)
print('Mean LL: {} - Two Stddev LL: {}'.format(mu_ll, sigma_ll))

# Just to make sure to switch to evaluation mode
model.eval()

# Draw some samples
n_samples = 10
samples = model.sample(n_samples ** 2).cpu()
images = torch.stack([inv_transform(x) for x in samples], dim=0)
torchvision.utils.save_image(images / 255.0, 'samples-cifar10.png', nrow=n_samples, padding=0)

# Save the model to file
torch.save(model.state_dict(), 'maf-cifar10.pt')
