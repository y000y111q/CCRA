# 文件：demo_ccra.py
import torch
from modules.ccra import CCRA


def main():
    batch_size = 2
    num_patches = 49   # 假装 7x7 的视觉 patch
    num_mem = 16       # 假装有 16 个“记忆槽”
    d_model = 512

    # 随机造两份特征
    att_feats = torch.randn(batch_size, num_patches, d_model)
    mem_feats = torch.randn(batch_size, num_mem, d_model)

    ccra = CCRA(d_model=d_model, nhead=8, num_layers=2)

    fused_att, fused_mem = ccra(att_feats, mem_feats)

    print("att_feats:", att_feats.shape)
    print("mem_feats:", mem_feats.shape)
    print("fused_att:", fused_att.shape)
    print("fused_mem:", fused_mem.shape)


if __name__ == "__main__":
    main()
