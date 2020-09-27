import torch
import torchvision

from spnflow.torch.models import RatSpn
from spnflow.torch.transforms import Flatten
from spnflow.torch.utils import torch_train, torch_test

n_features = 784
out_classes = 10

# Set the preprocessing transformation
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.1307, 0.3081),
    Flatten()
])

# Load the dataset
data_train = torchvision.datasets.MNIST('datasets', train=True, transform=transform, download=True)
data_test = torchvision.datasets.MNIST('datasets', train=False, transform=transform, download=True)
n_val = int(0.1 * len(data_train))
n_train = len(data_train) - n_val
data_train, data_val = torch.utils.data.random_split(data_train, [n_train, n_val])

# Build the model
model = RatSpn(
    n_features, out_classes,
    rg_depth=3,
    rg_repetitions=32,
    n_batch=16,
    n_sum=16,
    dropout=0.2,
    optimize_scale=False
)

# Train the model using discriminative setting (i.e. by minimizing the categorical cross-entropy)
torch_train(
    model, data_train, data_val,
    setting='discriminative',
    epochs=25,
    patience=5
)

# Test the model
nll, accuracy = torch_test(model, data_test, setting='discriminative')
print('Test NLL: %.4f' % nll)
print('Test Accuracy: %.1f' % (accuracy * 100))

# Save the model to file
torch.save(model.state_dict(), 'ratspn_mnist.pt')
