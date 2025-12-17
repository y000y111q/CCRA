#!/bin/bash
# train_iu_xray.sh - 训练 IU-Xray 数据集的脚本（稳妥版）

# 激活 base 环境
source /root/miniconda3/bin/activate base

# 确认使用的 Python 和 pip
echo "Using Python at: $(which python)"
python -V

#!/bin/bash
# 先确保脚本有执行权限
# chmod +x train_iu_xray.sh

python main_train.py \
--image_dir /root/autodl-fs/iu_xray/iu_xray/images \
--ann_path /root/autodl-fs/iu_xray/iu_xray/annotation.json \
--dataset_name iu_xray \
--max_seq_length 60 \
--threshold 3 \
--num_workers 2 \
--batch_size 16 \
--visual_extractor resnet101 \
--visual_extractor_pretrained True \
--d_model 512 \
--d_ff 512 \
--d_vf 2048 \
--num_heads 8 \
--num_layers 3 \
--dropout 0.1 \
--logit_layers 1 \
--bos_idx 0 \
--eos_idx 0 \
--pad_idx 0 \
--use_bn 0 \
--drop_prob_lm 0.5 \
--rm_num_slots 3 \
--rm_num_heads 8 \
--rm_d_model 512 \
--sample_method beam_search \
--beam_size 3 \
--temperature 1.0 \
--sample_n 1 \
--group_size 1 \
--output_logsoftmax 1 \
--decoding_constraint 0 \
--block_trigrams 1 \
--n_gpu 1 \
--epochs 100 \
--save_dir /root/autodl-fs/iu_xray/results \
--record_dir /root/autodl-fs/iu_xray/records \
--save_period 1 \
--monitor_mode max \
--monitor_metric BLEU_4 \
--early_stop 50 \
--optim Adam \
--lr_ve 5e-5 \
--lr_ed 1e-4 \
--weight_decay 5e-5 \
--amsgrad True \
--lr_scheduler StepLR \
--step_size 50 \
--gamma 0.1 \
--seed 9223
