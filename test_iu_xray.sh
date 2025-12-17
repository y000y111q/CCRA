#!/bin/bash

# 1. 激活环境
source /root/miniconda3/bin/activate ccra

# 2. 进到项目目录
cd /root/autodl-tmp/CCRA

# 3. 跑训练

python main_test.py \
--image_dir data/iu_xray/images/ \
--ann_path data/iu_xray/annotation.json \
--dataset_name iu_xray \
--max_seq_length 60 \
--threshold 3 \#没懂
--batch_size 16 \
--epochs 100 \
--save_dir results/iu_xray \
--step_size 50 \
--gamma 0.1 \
--seed 9223 \
--load data/model_iu_xray.pth
