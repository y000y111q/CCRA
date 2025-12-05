import torch
import torch.nn as nn
import torch.nn.functional as F


class MVSEM(nn.Module):
    """
    GRU简化版 MVSEM（多视图语义嵌入模块）：
    核心逻辑：使用 GRU 按照报告 token 序列的时间步进行记忆积累。
    同时集成了 Index 修复和 Non-linear Projection。

    输入: reports_ids [B, T]
    输出: MS [B, T, out_dim] (例如 [B, T, 2048])
    """

    def __init__(self, args, tokenizer):
        super(MVSEM, self).__init__()
        # --- 1. 维度配置 ---
        # 内部维度 d_model：用于 Embedding 和 GRU (默认 512)
        d_model = getattr(args, "d_model", 512)
        self.d_model = d_model
        # 输出维度 out_dim：外部需要的维度 (例如 CCRA 期望的 2048 视觉特征维度)
        self.out_dim = getattr(args, "d_vf", 2048)

        # --- 2. 核心组件初始化 ---

        # 🔴 词表大小判断 (采用最安全的方法: max index + 1，修复 IndexError)
        if hasattr(tokenizer, "vocab_size"):
            vocab_size = tokenizer.vocab_size
        elif hasattr(tokenizer, "token2idx"):
            # 找到最大的索引值，并 +1 确保 Embedding 空间足够
            vocab_size = max(tokenizer.token2idx.values()) + 1
        elif hasattr(tokenizer, "idx2token"):
            if isinstance(tokenizer.idx2token, dict):
                vocab_size = max(tokenizer.idx2token.keys()) + 1
            else:
                vocab_size = len(tokenizer.idx2token)
        else:
            raise ValueError(
                "Cannot infer vocab size from tokenizer; please check tokenizer attributes."
            )

        pad_id = getattr(tokenizer, "pad_token_id", 0)
        # A. 词嵌入层 (内部 d_model 维)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # B. GRU 单元 (实现按时间步更新记忆 M_t 的功能)
        self.gru = nn.GRU(
            input_size=self.d_model,  # 输入是 d_model 维的 embedding
            hidden_size=self.d_model,  # 隐藏状态 M_t 也是 d_model 维
            num_layers=1,
            batch_first=True  # 确保输入输出是 [B, T, D]
        )
        # --- 3. 非线性投影层 (512 -> 2048 对齐) ---
        if self.d_model != self.out_dim:
            # Non-linear Projection: Linear -> ReLU -> Dropout
            self.projector = nn.Sequential(
                nn.Linear(self.d_model, self.out_dim),  # 将 d_model (512) 投射到 out_dim (2048)
                nn.ReLU(),  # 非线性激活
                nn.Dropout(0.1)  # 防止过拟合
            )
        else:
            self.projector = None

    def forward(self, reports_ids):
        """
        reports_ids: [B, T]
        返回: MS: [B, T, out_dim] 
        """
        # 1. Token -> Embedding: [B, T, d_model]
        emb = self.embedding(reports_ids)
        # 2. GRU 更新记忆：
        #    MS 是 GRU 在每个时间步的隐藏状态 M_t，它就是论文中说的“串起来的 Mt”
        MS, _ = self.gru(emb)  # MS: [B, T, d_model]
        # 3. 应用非线性投影 [B, T, d_model] -> [B, T, out_dim] 
        if self.projector is not None:
            MS = self.projector(MS)
        return MS
