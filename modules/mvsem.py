import torch
import torch.nn as nn
import torch.nn.functional as F


class MVSEM(nn.Module):
    """
    Final MVSEM - 结合了以下特性:
    1. Memory Network (MHA 循环) 核心逻辑，符合顶刊论文设计思路。
    2. 安全的 vocab_size 计算 (max index + 1)，解决 IndexError。
    3. 非线性投影层 (512 -> 2048)，解决 CCRA 维度不匹配问题。

    输入: reports_ids [B, T]
    输出: MS [B, T, out_dim] (例如 [B, T, 2048])
    """

    def __init__(self, args, tokenizer):
        super(MVSEM, self).__init__()

        # --- 1. 维度配置 ---
        d_model = getattr(args, "d_model", 512)#记忆网络内部工作的维度（类似 Transformer 里面的 d_model），先用 512。
        self.d_model = d_model
        self.out_dim = getattr(args, "d_vf", 2048)  # 目标输出维度 (2048 for CCRA)
        self.num_memory = getattr(args, "num_memory", 8)#记忆槽的个数，比如 8 个“便签”。

        # --- 2. 核心组件初始化 ---

        # 词表大小判断 (🔴 采用最安全的方法: max index + 1)通过 tokenizer 的不同属性尝试推断 vocab_size：
        # 如果有 vocab_size，直接用。 # 否则用 token2idx 的最大值加 1，或 idx2token 的最大/长度来推断。
        # 这个策略确保 Embedding 的 vocab_size 至少覆盖输入中的最大 token 索引，避免 IndexError。
        if hasattr(tokenizer, "vocab_size"):
            vocab_size = tokenizer.vocab_size
        elif hasattr(tokenizer, "token2idx"):
            # 找到字典中最大的索引值，并 +1 确保 Embedding 空间足够
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

        pad_id = getattr(tokenizer, "pad_token_id", 0) #Embedding 时传给 padding_idx，确保填充 token 不参与梯度更新并在伪掩码中处理。

        # A. 词嵌入层 (内部 d_model 维)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)#输入 reports_ids 的形状为 [B, T]，嵌入输出为 [B, T, d_model]。

        # B. 初始记忆矩阵 M0 (可学习参数)
        self.mem_init = nn.Parameter(torch.randn(self.num_memory, d_model))

        # C. 记忆更新用的 Multi-Head Attention 层 (在 d_model 维度上操作)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=getattr(args, "num_heads", 8),
            batch_first=True
        )#使用 embed_dim=d_model，头数默认 8。batch_first=True 表示输入格式为 [B, S, E]，与后续代码保持一致。
        self.ln = nn.LayerNorm(d_model)  # 用于 MHA 后的 LayerNorm

        # --- 3. 非线性投影层 (512 -> 2048 对齐) ---
        if self.d_model != self.out_dim:
            # Non-linear Projection: 符合论文要求的特征对齐
            self.projector = nn.Sequential(
                nn.Linear(self.d_model, self.out_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            self.projector = None

    def forward(self, reports_ids):
        """
        reports_ids: [B, T]
        返回: MS: [B, T, out_dim]
        """
        device = reports_ids.device
        B, T = reports_ids.shape

        # 1. Token -> Embedding: [B, T, d_model]
        emb = self.embedding(reports_ids)

        # 2. 初始化记忆 M0 -> M_t: [B, N_M, d_model]
        M_t = self.mem_init.unsqueeze(0).expand(B, self.num_memory, self.d_model).to(device)

        mem_seq = []

        # 3. 循环每个时间步 t，执行 Memory Update (论文的核心逻辑)
        for t in range(T):
            # 当前词向量 y_t: [B, d_model]
            y_t = emb[:, t, :]

            # 拼接: Key/Value = [Memory, Current_Word] -> [B, N_M+1, d_model]
            kv = torch.cat([M_t, y_t.unsqueeze(1)], dim=1)

            # MHA 更新记忆：
            M_new, _ = self.mha(query=M_t, key=kv, value=kv)

            # 残差连接 + LayerNorm
            M_t = self.ln(M_t + M_new)

            # 摘要: 对 N_M 个记忆槽取平均，得到当前时间步的上下文向量 M_t -> [B, d_model]
            mem_summary = M_t.mean(dim=1)
            mem_seq.append(mem_summary)

        # 4. 堆叠得到原始记忆流 MS [B, T, d_model]
        MS = torch.stack(mem_seq, dim=1)

        # 5. 应用非线性投影 [B, T, d_model] -> [B, T, out_dim] (对齐CCRA)
        if self.projector is not None:
            MS = self.projector(MS)

        return MS
