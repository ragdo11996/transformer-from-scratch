#!/usr/bin/env bash
# usage: bash scripts/run.sh
set -e

# create env
# conda create -n transformer python=3.10 -y
# pip install -r requirements.txt

CONFIG=configs/base.yaml
SEED=42
DEVICE=cuda

python src/train.py --config $CONFIG --seed $SEED --device $DEVICE
