#!/bin/bash
export PYTHONPATH=../../
export DATAPATH=../
python experiment.py miniboone
python experiment.py hepmass
python experiment.py power
python experiment.py gas
python experiment.py bsds300
python experiment.py mnist
python experiment.py cifar10
