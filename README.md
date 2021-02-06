![Logo](sphinx/_static/spnflow-logo.svg)

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![PyPI version](https://badge.fury.io/py/spnflow.svg)](https://badge.fury.io/py/spnflow)

# SPNFlow

## Abstract
SPNFlow is a Python library that implements probabilistic models such as various kinds of **Sum-Product Networks**,
**Normalizing Flows** and their possible combinations for tractable probabilistic inference.
Some models are implemented using **PyTorch** for fast training and inference on GPUs.

## Features
- Inference algorithms for SPNs. [1] [2] [4]
- Learning algorithms for SPNs structure. [1] [2] [3] [4]
- Optimization of the structure of SPNs. [4]
- JSON I/O operations for SPNs. [4]
- Implementation of RAT-SPN using PyTorch. [5]
- Implementation of MAFs and Real-NVPs using PyTorch. [6] [7] [8]
- Implementation of Deep Generalized Convolutional SPNs (DGC-SPNs). [9]

## Documentation
The library documentation is hosted using Github Pages at [SPNFlow](https://loreloc.github.io/spnflow/).

## Experiments
The datasets required to run the experiments can be found on [Google Drive](https://drive.google.com/file/d/1iVBjQts_8aADdXYEFFhpxfQgz2k5tU7f/view?usp=sharing).
After downloading it, unzip it in `experiments/datasets` to be able to run the experiments.

## Examples
Various code examples can be found in `examples` directory.

## Related Repositories
- [SPFlow](https://github.com/SPFlow/SPFlow)
- [RAT-SPN](https://github.com/cambridge-mlg/RAT-SPN)
- [MAF](https://github.com/gpapamak/maf)
- [LibSPN-Keras](https://github.com/pronobis/libspn-keras)

## References
1. On Theoretical Properties of Sum-Product Networks (Peharz et al.).
2. Sum-Product Networks: A New Deep Architecture (Poon and Domingos).
3. Mixed Sum-Product Networks: A Deep Architecture for Hybrid Domains (Molina, Vergari et al.).
4. SPFLOW : An easy and extensible library for deep probabilistic learning using Sum-Product Networks (Molina, Vergari et al.).
5. Probabilistic Deep Learning using Random Sum-Product Networks (Peharz et al.).
6. Masked Autoregressive Flow for Density Estimation (Papamakarios et al.).
7. Density Estimation using RealNVP (Dinh et al.).
8. Normalizing Flows for Probabilistic Modeling and Inference (Papamakarios, Nalisnick et al.).
9. Deep Generalized Convolutional Sum-Product Networks for Probabilistic Image Representations (Van de Wolfshaar and Pronobis).
