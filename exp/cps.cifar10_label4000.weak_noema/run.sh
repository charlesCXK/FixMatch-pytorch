#!/usr/bin/env bash
#export NGPUS=1
#export CUDA_VISIBLE_DEVICES=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py

nvidia-smi
source activate base

export cxk_volna=$1"/"
export snapshot_iter=1

export learning_rate=0.03
export batch_size=16 # 16*4
export NGPUS=4
export DATASET=cifar10
export LABEL=4000

python -m torch.distributed.launch --nproc_per_node $NGPUS ./train.py \
--dataset $DATASET --num-labeled $LABEL --arch wideresnet --batch-size $batch_size --lr $learning_rate \
--expand-labels --seed 5 --out results/cifar10@4000 --use-ema False

#python -m torch.distributed.launch --nproc_per_node 1 ./train.py \
#--dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 4 --lr 0.03 \
#--expand-labels --seed 5 --out results/cifar10@4000