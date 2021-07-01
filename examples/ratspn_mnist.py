import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from deeprob.spn.models.ratspn import GaussianRatSpn
from deeprob.torch.transforms import Flatten
from deeprob.torch.routines import train_model, test_model

n_features = 784
out_classes = 10

# Set the preprocessing transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    Flatten()
])

# Load the dataset
data_train = datasets.MNIST('datasets', train=True, transform=transform, download=True)
data_test = datasets.MNIST('datasets', train=False, transform=transform, download=True)
n_val = int(0.1 * len(data_train))
n_train = len(data_train) - n_val
data_train, data_val = data.random_split(data_train, [n_train, n_val])

# Build the model
model = GaussianRatSpn(
    n_features,
    out_classes=out_classes,
    rg_depth=2,
    rg_repetitions=16,
    rg_batch=16,
    rg_sum=16,
    sum_dropout=0.2,
    optimize_scale=False
)

# Train the model using discriminative setting (i.e. by minimizing the categorical cross-entropy)
train_model(
    model, data_train, data_val,
    setting='discriminative',
    lr=1e-2,
    epochs=25,
    patience=5
)

# Test the model
nll, metrics = test_model(model, data_test, setting='discriminative')
print('Test NLL: {:.4f}'.format(nll))
print('Test Metrics: {}'.format(metrics))

# Save the model to file
torch.save(model.state_dict(), 'ratspn-mnist.pt')
