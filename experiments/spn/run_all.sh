#!/bin/bash
export PYTHONPATH=../../
export DATAROOT=../datasets/
python experiment.py hepmass
python experiment.py miniboone
python experiment.py power
python experiment.py gas
python experiment.py bsds300
