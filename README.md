# SPNFlow
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract
Sum-product networks (SPNs) are models that, under some constraints, guarantee tractability for probabilistic inference and sampling.
Recent works show that it's possible to treat SPNs as multiple layers neural models with density distributions as input units and log-likelihoods as outputs.
A lot of work have been done on normalizing flows as density estimators.
Like sum-product networks, some normalizing flows guarantee tractability and have been successfully used to build efficient models for probabilistic inference.
In this work I exploit the potentiality of normalizing flows combined with SPNs for both tractable and expressive density estimation.

## Introduction
**SPNFlow** is a python implementation of **Sum-Product Networks** (SPNs) learning and inference.
It also contains an implementation of **Randomized and Tensorized SPNs** (RAT-SPNs) using Tensorflow 2.0.
In this repository there are also present novel models that combine **RAT-SPNs** and **Normalizing Flows** for tractable density estimation.

## Features
- Inference algorithms for SPNs. [1] [2] [4]
- Learning algorithms for SPNs structure. [1] [2] [3] [4]
- Optimization of the structure of SPNs. [4]
- JSON I/O operations for SPNs. [4]
- Implementation of RAT-SPN using Keras and Tensorflow 2.0. [5]
- Implementation of RAT-SPN combined with MAFs and RealNVP using Keras and Tensorflow 2.0. [5] [6] [7] [8]

## Documentation
The online documentation is hosted using Github Pages at [SPNFlow](https://loreloc.github.io/spnflow/).

## Related Repositories
- [SPFlow](https://github.com/SPFlow/SPFlow)
- [RAT-SPN](https://github.com/cambridge-mlg/RAT-SPN)
- [MAF](https://github.com/gpapamak/maf)

## References
1. On Theoretical Properties of Sum-Product Networks (Peharz et al.).
2. Sum-Product Networks: A New Deep Architecture (Poon and Domingos).
3. Mixed Sum-Product Networks: A Deep Architecture for Hybrid Domains (Molina, Vergari et al.).
4. SPFLOW : An easy and extensible library for deep probabilistic learning using Sum-Product Networks (Molina, Vergari et al.).
5. Probabilistic Deep Learning using Random Sum-Product Networks (Peharz et al.).
6. Masked Autoregressive Flow for Density Estimation (Papamakarios et al.).
7. Density Estimation using RealNVP (Dinh et al.).
8. Normalizing Flows for Probabilistic Modeling and Inference (Papamakarios, Nalisnick et al.).
