import numpy as np
import sklearn as sk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from deeprob.spn.utils.data import DataStandardizer
from deeprob.spn.utils.statistics import get_statistics
from deeprob.spn.structure.leaf import Gaussian, Categorical
from deeprob.spn.learning.wrappers import learn_classifier
from deeprob.spn.algorithms.inference import mpe
from deeprob.spn.algorithms.sampling import sample
from deeprob.spn.structure.io import save_json

# Load the MNIST dataset
n_classes = 10
image_w, image_h = 28, 28
transform = transforms.ToTensor()
data_train = datasets.MNIST('datasets', train=True, transform=transform, download=True)
data_test = datasets.MNIST('datasets', train=False, transform=transform, download=True)

# Build the autoencoder for features extraction
n_features = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, n_features),
    nn.ReLU(),
)
decoder = nn.Sequential(
    nn.Linear(n_features, 256),
    nn.ReLU(),
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 784),
    nn.Sigmoid(),
    nn.Unflatten(1, (1, 28, 28))
)
autoencoder = nn.Sequential(encoder, decoder)
autoencoder.to(device)

# Train the autoencoder
epochs = 20
batch_size = 128
lr = 1e-3
train_loader = data.DataLoader(data_train, batch_size=batch_size)
optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
criterion = nn.BCELoss()
for epoch in range(epochs):
    train_loss = 0.0
    for (inputs, labels) in train_loader:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_size
    train_loss /= len(train_loader)
    print('Epoch: {} Train loss: {}'.format(epoch + 1, train_loss))

# Extract the features from the dataset
x_train = np.zeros([len(data_train), n_features])
y_train = np.zeros(len(data_train))
for i in range(len(data_train)):
    input, label = data_train[i]
    inputs = torch.unsqueeze(input, dim=0).to(device)
    outputs = encoder(inputs).cpu().detach().numpy()
    x_train[i] = outputs[0]
    y_train[i] = label
x_test = np.zeros([len(data_test), n_features])
y_test = np.zeros(len(data_test))
for i in range(len(data_test)):
    input, label = data_test[i]
    inputs = torch.unsqueeze(input, dim=0).to(device)
    outputs = encoder(inputs).cpu().detach().numpy()
    x_test[i] = outputs[0]
    y_test[i] = label

# Preprocess the dataset
transform = DataStandardizer()
transform.fit(x_train)
x_train = transform.forward(x_train)
x_test = transform.forward(x_test)

# Learn the SPN structure and parameters, as a classifier
distributions = [Gaussian] * n_features + [Categorical]
data_train = np.column_stack([x_train, y_train])
spn = learn_classifier(
    data_train,
    distributions,
    learn_leaf='mle',
    split_rows='kmeans',
    split_cols='rdc',
    min_rows_slice=200,
    split_rows_kwargs={'n': 2},
    split_cols_kwargs={'d': 0.3}
)

# Print some statistics
print(get_statistics(spn))

# Save the model to a JSON file
save_json(spn, 'spn-mnist.json')

# Make some predictions on the test set classes
nans = np.tile(np.nan, [len(x_test), 1])
data_test = np.column_stack([x_test, nans])
y_pred = mpe(spn, data_test)[:, -1]

# Plot a summary of the classification
print(sk.metrics.classification_report(y_test, y_pred))

# Make some sampling for each class
n_samples = 10
nans = np.tile(np.nan, [n_samples * n_classes, n_features])
classes = np.tile(np.arange(n_classes), [1, n_samples]).T
data_samples = np.column_stack([nans, classes])
data_samples = sample(spn, data_samples)[:, :-1]

# Transform back the data and plot the samples
data_images = transform.backward(data_samples)
inputs = torch.tensor(data_images, dtype=torch.float32, device=device)
data_images = decoder(inputs).cpu().detach().numpy()
data_images = data_images.reshape(n_samples * n_classes, 1, image_h, image_w)
torchvision.utils.save_image(torch.tensor(data_images), 'samples-mnist.png', nrow=n_samples, padding=0)
