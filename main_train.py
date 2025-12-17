import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.r2gen import R2GenModel
from torch.optim.lr_scheduler import StepLR, LambdaLR


def parse_args():
    parser = argparse.ArgumentParser()

    # ----------------- 数据设置 -----------------
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json')
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'])
    parser.add_argument('--max_seq_length', type=int, default=60)
    parser.add_argument('--threshold', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)

    # ----------------- 模型设置 -----------------
    parser.add_argument('--visual_extractor', type=str, default='resnet101')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--d_vf', type=int, default=2048)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--logit_layers', type=int, default=1)
    parser.add_argument('--bos_idx', type=int, default=0)
    parser.add_argument('--eos_idx', type=int, default=0)
    parser.add_argument('--pad_idx', type=int, default=0)
    parser.add_argument('--use_bn', type=int, default=0)
    parser.add_argument('--drop_prob_lm', type=float, default=0.5)
    parser.add_argument('--rm_num_slots', type=int, default=3)
    parser.add_argument('--rm_num_heads', type=int, default=8)
    parser.add_argument('--rm_d_model', type=int, default=512)

    # ----------------- 报告生成采样 -----------------
    parser.add_argument('--sample_method', type=str, default='beam_search')
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sample_n', type=int, default=1)
    parser.add_argument('--group_size', type=int, default=1)
    parser.add_argument('--output_logsoftmax', type=int, default=1)
    parser.add_argument('--decoding_constraint', type=int, default=0)
    parser.add_argument('--block_trigrams', type=int, default=1)

    # ----------------- 训练器设置 -----------------
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='results/iu_xray')
    parser.add_argument('--record_dir', type=str, default='records/')
    parser.add_argument('--save_period', type=int, default=1)
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'])
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4')
    parser.add_argument('--early_stop', type=int, default=50)

    # ----------------- 优化器 -----------------
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr_ve', type=float, default=5e-5)
    parser.add_argument('--lr_ed', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--amsgrad', type=bool, default=True)

    # ----------------- 学习率调度器 -----------------
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', choices=['StepLR', 'LambdaLR'])
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.1)

    # ----------------- 其他 -----------------
    parser.add_argument('--seed', type=int, default=9233)
    parser.add_argument('--resume', type=str, help='checkpoint path')

    args = parser.parse_args()
    return args


def build_lr_scheduler_custom(args, optimizer):
    """支持 StepLR 和 LambdaLR"""
    if args.lr_scheduler == 'StepLR':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'LambdaLR':
        # 简单指数衰减示例
        lambda_fn = lambda epoch: 0.95 ** epoch
        scheduler = LambdaLR(optimizer, lr_lambda=lambda_fn)
    else:
        raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")
    return scheduler


def main():
    args = parse_args()

    # 固定随机种子
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # 初始化 tokenizer
    tokenizer = Tokenizer(args)

    # 数据加载
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # 构建模型
    model = R2GenModel(args, tokenizer)

    # 损失和指标
    criterion = compute_loss
    metrics = compute_scores

    # 优化器和学习率调度
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler_custom(args, optimizer)

    # 训练器
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler,
                      train_dataloader, val_dataloader, test_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()
