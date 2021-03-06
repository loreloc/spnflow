import torch
import torchvision

from deeprob.torch.models.dgcspn import DgcSpn
from deeprob.torch.transforms import Reshape
from deeprob.torch.routines import train_model, test_model

image_size = (1, 28, 28)
n_classes = 10

# Set the preprocessing transformation
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    Reshape(image_size)
])

# Load the dataset
data_train = torchvision.datasets.MNIST('datasets', train=True, transform=transform, download=True)
data_test = torchvision.datasets.MNIST('datasets', train=False, transform=transform, download=True)
n_val = int(0.1 * len(data_train))
n_train = len(data_train) - n_val
data_train, data_val = torch.utils.data.random_split(data_train, [n_train, n_val])

# Instantiate the model
model = DgcSpn(
    in_size=image_size,
    out_classes=n_classes,
    n_batch=16,
    sum_channels=32,
    depthwise=True,
    n_pooling=2,
    optimize_scale=False,
    sum_dropout=0.2,
    uniform_loc=(-1.5, 1.5)
)

# Train the model using discriminative setting (i.e. by minimizing the categorical cross-entropy)
train_model(
    model, data_train, data_val,
    setting='discriminative',
    lr=1e-2,
    batch_size=64,
    epochs=10
)

# Test the model
nll, metrics = test_model(model, data_test, setting='discriminative')
print('Test NLL: {:.4f}'.format(nll))
print('Test Metrics: {}'.format(metrics))

# Save the model to file
torch.save(model.state_dict(), 'dgcspn-mnist.pt')
