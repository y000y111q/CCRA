import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
from modules.ccra import CCRA
from modules.mvsem import MVSEM


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()#：把 PyTorch 父类的初始化先做了，这是固定写法。
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        # ⭐ 新增：MVSEM（极简版，先把文本 → 向量序列）
        self.mvsem = MVSEM(args, tokenizer)
        # Phase 1: CCRA 的特征维度要和 att_feats 一致（VisualExtractor 输出是 2048 维）
        self.ccra = CCRA(
            d_model=2048,   # ← 关键就这一行，从 512 改成 2048
            nhead=8,        # 2048 / 8 = 256，每个 head 256 维，合理
            num_layers=4,
            dim_ff=4096,    # 前馈层维度，先用 2 倍 2048，别太大，CPU 也扛得住
            dropout=0.1,
        )
        # MVSEM 输出是 d_model=512 维，视觉特征是 d_vf=2048 维，
        # 用这个线性层把 MS 从 512 投影到 2048，便于送入 CCRA。
        self.ms_proj = nn.Linear(args.d_model, args.d_vf)


#（这些超参数以后可以根据论文微调，现在先用一个稳妥配置。）
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):
        """
        images: [B, 2, 3, H, W]
        targets: [B, T]
        """
        # 1. ResNet提取特征 (输出是 2048 维)
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])

        # 拼接特征
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)  # [B, 2L, 2048]

        # 2. 训练阶段：记忆增强
        if mode == 'train' and targets is not None:
            # === MVSEM 模块 ===
            # 因为我们在 MVSEM 内部加了投影，这里出来的 MS 直接就是 [B, T, 2048]
            # 不需要再写 self.ms_proj(MS) 了！代码更干净！
            MS = self.mvsem(targets)

            # === CCRA 模块 ===
            # 输入两个都是 2048 维，完美匹配
            att_feats, _ = self.ccra(att_feats, MS)

        else:
            # 验证/测试/无Target阶段
            att_feats, _ = self.ccra(att_feats, att_feats)

        # 3. 生成报告
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError

        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

