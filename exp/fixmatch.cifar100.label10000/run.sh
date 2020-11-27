#!/usr/bin/env bash
#export NGPUS=1
#export CUDA_VISIBLE_DEVICES=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py

nvidia-smi
source activate base

export cxk_volna=$1"/"
export snapshot_iter=1

export learning_rate=0.03
export nepochs=80
export batch_size=16 # 16*4
export NGPUS=4
export DATASET=cifar100
export LABEL=10000

python -m torch.distributed.launch --nproc_per_node $NGPUS ./train.py \
--dataset $DATASET --num-labeled $LABEL --arch wideresnet --batch-size $batch_size --lr $learning_rate  --wdecay 0.001 \
--expand-labels --seed 5 --out results/$DATASET@$LABEL


#python -m torch.distributed.launch --nproc_per_node 1 ./train.py \
#--dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 4 --lr 0.03 \
#--expand-labels --seed 5 --out results/cifar100@10000
