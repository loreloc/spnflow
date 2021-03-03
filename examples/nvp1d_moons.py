import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from spnflow.torch.models.flows import RealNVP1d
from spnflow.torch.routines import torch_train


x_train, y_train = make_moons(n_samples=10000, shuffle=True, noise=0.05)
x_train = x_train.astype(np.float32)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

plt.scatter(x_train[:, 0], x_train[:, 1], marker='o', s=2)
plt.show()

model = RealNVP1d(in_features=2, n_flows=5, depth=2, units=128, batch_norm=False)
torch_train(
    model, x_train, x_valid,
    setting='generative', lr=1e-4,
    batch_size=100, epochs=100, patience=20
)

samples = model.sample(2000).cpu().numpy()
plt.scatter(samples[:, 0], samples[:, 1], marker='o', s=2)
plt.show()
