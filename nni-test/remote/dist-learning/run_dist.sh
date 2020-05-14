#/bin/bash

#export TF_CONFIG="$(cat $1)"

export TF_CONFIG='{"cluster": {"ps": ["xxxx"], "worker": ["xxxx", "xxxx"]}, "task": {"type": "worker", "index": 1}}'
export DATADIR="./data/"

mkdir -p $DATADIR

python mnist.py --data_dir=$DATADIR --num_gpus=0 --train_steps=100