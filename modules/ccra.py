# 文件：modules/ccra.py

import torch
import torch.nn as nn


class CCRABlock(nn.Module):
    """
    一个简单的 CCRA 积木：
    - 输入：
        major: [B, Lm, D]  主要序列（比如图像 patch 特征）
        supp:  [B, Ls, D]  辅助序列（比如 memory 特征；Phase 1 可以先用同一个特征代替）
    - 输出：
        major_new: 融合了辅助信息后的主要序列
        supp_new:  融合了主要信息后的辅助序列（先留着，后面 MVSEM 可以用）
    """
    def __init__(self, d_model=512, nhead=8, dim_ff=2048, dropout=0.1):
        super().__init__()

        # supp 看 major
        self.attn_supp_to_major = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,  # 习惯用 [B, L, D]
        )
        # major 看 supp
        self.attn_major_to_supp = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # FFN进行残差连接和归一化，用于特征增强。
        self.ffn_major = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.ffn_supp = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )

        # LayerNorm + Dropout进行残差连接和归一化，用于特征增强。
        self.norm_major1 = nn.LayerNorm(d_model)
        self.norm_major2 = nn.LayerNorm(d_model)
        self.norm_supp1 = nn.LayerNorm(d_model)
        self.norm_supp2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, major, supp, major_mask=None, supp_mask=None):
        """
        major: [B, Lm, D]
        supp:  [B, Ls, D]
        mask 先不认真用，预留接口
        """
        # 1) supp 去看 major（major 做 key/value）
        attn_out_supp, _ = self.attn_supp_to_major(
            query=supp,
            key=major,
            value=major,
            key_padding_mask=major_mask,  # 可以为 None
        )
        supp2 = self.norm_supp1(supp + self.dropout(attn_out_supp))
        supp2 = self.norm_supp2(supp2 + self.dropout(self.ffn_supp(supp2)))

        # 2) major 再去看更新后的 supp
        attn_out_major, _ = self.attn_major_to_supp(
            query=major,
            key=supp2,
            value=supp2,
            key_padding_mask=supp_mask,
        )
        major2 = self.norm_major1(major + self.dropout(attn_out_major))
        major2 = self.norm_major2(major2 + self.dropout(self.ffn_major(major2)))

        return major2, supp2


class CCRA(nn.Module):
    """
    堆几层 CCRABlock，得到最终融合后的 major / supp
    Phase 1：我们主要用 fused_major 喂给 Decoder。
    实现了论文中 Repeat（重复） 的机制，让视觉特征和语义记忆在多层中反复交错、清洗和融合。
    """
    def __init__(self, d_model=512, nhead=8, num_layers=2, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CCRABlock(d_model=d_model, nhead=nhead, dim_ff=dim_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, major, supp, major_mask=None, supp_mask=None):
        """
        major: [B, Lm, D]
        supp:  [B, Ls, D]
        """
        for layer in self.layers:
            major, supp = layer(major, supp, major_mask, supp_mask)
        return major, supp
