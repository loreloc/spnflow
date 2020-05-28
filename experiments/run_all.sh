#!/bin/bash
export PYTHONPATH=../
CUDA_VISIBLE_DEVICES="" python experiment.py power
CUDA_VISIBLE_DEVICES="" python experiment.py gas
python experiment.py hepmass
python experiment.py miniboone
python experiment.py bsds300
python experiment.py mnist
python experiment.py cifar10
