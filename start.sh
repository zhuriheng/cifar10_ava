#!/bin/bash
set -eux
set -o pipefail

echo "downloading example mxnet resnet50 model"
mkdir -p /workspace/model
cd /workspace/model
wget http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50-0000.params
wget http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50-symbol.json

echo "start to run training script"
python /workspace/examples/trainings/mxnet/training.py
python /workspace/runs/projects/training.py
