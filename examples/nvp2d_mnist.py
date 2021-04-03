import torch
import torchvision

from spnflow.torch.models.flows import RealNVP2d
from spnflow.torch.transforms import Dequantize
from spnflow.torch.routines import torch_train


class UnsupervisedMNIST(torchvision.datasets.MNIST):
    N_FEATURES = 784
    IMAGE_SIZE = (1, 28, 28)

    def __init__(self, root, train=True, transform=None, download=False):
        super(UnsupervisedMNIST, self).__init__(root, train, transform, download=download)

    def __getitem__(self, index):
        x, y = super(UnsupervisedMNIST, self).__getitem__(index)
        return x


# Set the preprocessing transformation
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    Dequantize()
])

# Load the MNIST dataset (unsupervised wrapping) and
# split the dataset in train set and validation set
in_features = UnsupervisedMNIST.IMAGE_SIZE
data_train = UnsupervisedMNIST('datasets', train=True, transform=transform, download=True)
n_val = int(0.1 * len(data_train))
n_train = len(data_train) - n_val
data_train, data_val = torch.utils.data.random_split(data_train, [n_train, n_val])

# Instantiate the model
model = RealNVP2d(
    in_features=in_features,
    logit=1e-6,
    n_flows=2,
    n_blocks=2,
    channels=32
)

# Train the model using generative setting (i.e. by maximizing the log-likelihood)
torch_train(
    model, data_train, data_val,
    setting='generative',
    batch_size=64,
    weight_decay=5e-5,
    epochs=20,
    patience=3
)

# Just to make sure to switch to evaluation mode
model.eval()

# Draw some samples
n_samples = 10
samples = model.sample(n_samples ** 2).cpu()
torchvision.utils.save_image(samples, 'samples_mnist.png', nrow=n_samples, padding=0)

# Save the model to file
torch.save(model.state_dict(), 'nvp_mnist.pt')
