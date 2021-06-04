import numpy as np
import matplotlib.pyplot as plt

from spnflow.structure.leaf import Gaussian
from spnflow.learning.wrappers import learn_estimator
from spnflow.algorithms.sampling import sample

from spnflow.torch.models.ratspn import GaussianRatSpn
from spnflow.torch.models.flows import RealNVP1d
from spnflow.torch.models.flows import MAF
from spnflow.torch.routines import train

LDOM, RDOM = -4.0, 4.0
RESOLUTION = 128
N_TRAIN_SAMPLES = 250_000
N_GEN_SAMPLES = 100_000


def energy_domain():
    x = np.linspace(LDOM, RDOM, num=RESOLUTION)
    y = np.linspace(LDOM, RDOM, num=RESOLUTION)
    x, y = np.meshgrid(x, y)
    return np.column_stack([x.flatten(), y.flatten()])


def w1(z):
    return np.sin(2.0 * np.pi * z[:, 0] / 4.0)


def w2(z):
    return 3.0 * np.exp(-0.5 * ((z[:, 0] - 1.0) / 0.6) ** 2)


def w3(z):
    return 3.0 / (1.0 + np.exp((1.0 - z[:, 0]) / 0.3))


def moons_pdf():
    z = energy_domain()
    u = 0.5 * ((np.linalg.norm(z, axis=1) - 2.0) / 0.4) ** 2
    v = np.exp(-0.5 * ((z[:, 0] - 2.0) / 0.6) ** 2)
    w = np.exp(-0.5 * ((z[:, 0] + 2.0) / 0.6) ** 2)
    p = np.exp(-u) * (v + w)
    return (p / np.sum(p)).reshape(RESOLUTION, RESOLUTION)


def waves_pdf():
    z = energy_domain()
    p = np.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2)
    return (p / np.sum(p)).reshape(RESOLUTION, RESOLUTION)


def split_pdf():
    z = energy_domain()
    u = np.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.35) ** 2)
    v = np.exp(-0.5 * ((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2)
    p = u + v
    return (p / np.sum(p)).reshape(RESOLUTION, RESOLUTION)


def nails_pdf():
    z = energy_domain()
    u = np.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2)
    v = np.exp(-0.5 * ((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2)
    p = u + v
    return (p / np.sum(p)).reshape(RESOLUTION, RESOLUTION)


def pdf_train_sample(pdf, noise=0.05):
    idx = np.random.choice(pdf.size, size=N_TRAIN_SAMPLES, p=pdf.ravel())
    x, y = np.unravel_index(idx, shape=pdf.shape)
    x = LDOM + (RDOM - LDOM) * (x / RESOLUTION)
    y = LDOM + (RDOM - LDOM) * (y / RESOLUTION)
    x = x + np.random.randn(N_TRAIN_SAMPLES) * noise
    y = y + np.random.randn(N_TRAIN_SAMPLES) * noise
    return np.column_stack([x, y]).astype(np.float32)


def energy_plot_pdf(ax, pdf):
    print()
    ax.imshow(pdf, interpolation='bicubic', cmap='jet')
    ax.axis('off')


def energy_plot_samples(ax, samples):
    pdf, _, _ = np.histogram2d(
        samples[:, 0], samples[:, 1],
        bins=RESOLUTION,
        range=[[LDOM, RDOM], [LDOM, RDOM]],
        density=True
    )
    energy_plot_pdf(ax, pdf)


def spn_sample_energy(data):
    distributions = [Gaussian, Gaussian]
    spn = learn_estimator(
        data, distributions,
        learn_leaf='mle',
        split_rows='gmm',
        split_cols='gvs',
        min_rows_slice=200,
        split_rows_kwargs={'n': 2},
        split_cols_kwargs={'p': 5.0}
    )
    nans = np.tile(np.nan, [N_GEN_SAMPLES, 2])
    return sample(spn, nans)


def rat_sample_energy(data):
    n_train = int(0.9 * len(data))
    data_valid = data[n_train:]
    data_train = data[:n_train]
    model = GaussianRatSpn(in_features=2, rg_depth=1, rg_repetitions=5, rg_batch=10, rg_sum=10)
    train(
        model, data_train, data_valid,
        setting='generative', lr=1e-3,
        batch_size=256, epochs=100, patience=1
    )
    model.eval()
    return model.sample(N_GEN_SAMPLES).cpu().numpy()


def nvp_sample_energy(data):
    n_train = int(0.9 * len(data))
    data_valid = data[n_train:]
    data_train = data[:n_train]
    model = RealNVP1d(in_features=2, n_flows=10, depth=1, units=256, batch_norm=False)
    train(
        model, data_train, data_valid,
        setting='generative', lr=1e-4,
        batch_size=256, epochs=100, patience=1
    )
    model.eval()
    return model.sample(N_GEN_SAMPLES).cpu().numpy()


def maf_sample_energy(data):
    n_train = int(0.9 * len(data))
    data_valid = data[n_train:]
    data_train = data[:n_train]
    model = MAF(in_features=2, n_flows=10, depth=1, units=256, batch_norm=False)
    train(
        model, data_train, data_valid,
        setting='generative', lr=1e-4,
        batch_size=256, epochs=100, patience=1
    )
    model.eval()
    return model.sample(N_GEN_SAMPLES).cpu().numpy()


if __name__ == '__main__':
    # Compute the PDFs
    pdfs = {
        'moons': moons_pdf(),
        'waves': waves_pdf(),
        'split': split_pdf(),
        'nails': nails_pdf()
    }

    # Get some samples
    samples = {
        'moons': pdf_train_sample(pdfs['moons']),
        'waves': pdf_train_sample(pdfs['waves']),
        'split': pdf_train_sample(pdfs['split']),
        'nails': pdf_train_sample(pdfs['nails'])
    }

    # Initialize the result plot
    fig, axs = plt.subplots(ncols=5, nrows=4, figsize=(20, 16))

    # Learn a model for each density function
    for i, energy in enumerate(samples.keys()):
        print('Model: SPN - Energy: ' + energy)
        energy_plot_samples(axs[i, 0], spn_sample_energy(samples[energy]))
        axs[0, 0].set_title('SPN', fontdict={'fontsize': 32})

        print('Model: RAT-SPN - Energy: ' + energy)
        energy_plot_samples(axs[i, 1], rat_sample_energy(samples[energy]))
        axs[0, 1].set_title('RAT-SPN', fontdict={'fontsize': 32})

        print('Model: RealNVP - Energy: ' + energy)
        energy_plot_samples(axs[i, 2], nvp_sample_energy(samples[energy]))
        axs[0, 2].set_title('RealNVP', fontdict={'fontsize': 32})

        print('Model: MAF - Energy: ' + energy)
        energy_plot_samples(axs[i, 3], maf_sample_energy(samples[energy]))
        axs[0, 3].set_title('MAF', fontdict={'fontsize': 32})

    # Plot the true density functions
    for i, energy in enumerate(pdfs.keys()):
        energy_plot_pdf(axs[i, -1], pdfs[energy])
        axs[0, -1].set_title('True', fontdict={'fontsize': 32})

    # Save the results
    plt.tight_layout()
    plt.savefig('energy.png')
