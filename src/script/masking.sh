#!/bin/bash
CHECKPOINT_PATH="../resource/checkpoint/masking/checkpoint.pth"

python pretrain_masking.py 
python finetune_linear.py --checkpoint_path $CHECKPOINT_PATH
python finetune_full.py --subsample_ratio 0.1 --checkpoint_path $CHECKPOINT_PATH
python finetune_full.py --subsample_ratio 0.01 --checkpoint_path $CHECKPOINT_PATH
