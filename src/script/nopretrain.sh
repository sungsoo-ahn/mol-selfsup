#!/bin/bash

CHECKPOINT_PATH="../resource/checkpoint/nopretrain/checkpoint.pth"

python pretrain_nopretrain.py
#python finetune_linear.py --checkpoint_path $CHECKPOINT_PATH
#python finetune_full.py --subsample_ratio 0.1 --checkpoint_path $CHECKPOINT_PATH
python finetune_full.py --subsample_ratio 0.01 --checkpoint_path $CHECKPOINT_PATH
