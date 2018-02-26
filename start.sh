#!/bin/bash
set -eux
set -o pipefail

echo "downloading example mxnet resnet50 model"
mkdir -p /workspace/model
cd /workspace/model
wget http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50-0000.params
wget http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50-symbol.json

echo "move the file default.json to the folder /workspace/params"
mkdir /workspace/params
mv default.json /workspace/params

echo "start to run training script"
python /workspace/runs/cifar10_ava/training.py
