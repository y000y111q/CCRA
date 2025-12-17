import os
import json
import torch
import sys
sys.path.append("/root/autodl-fs/ccra_runs/iu_xray_mvsem_202512/")
from dataset import IuxrayMultiImageDataset

from torch.utils.data import DataLoader

from model import R2GenModel
from transformers import AutoTokenizer
from torchvision import transforms

# COCO eval
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# -------------------------------
# 配置
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 2
image_dir = "/root/autodl-fs/iu_xray/images"
ann_path = "/root/autodl-fs/iu_xray/annotation.json"
checkpoint_path = "/root/autodl-fs/ccra_runs/iu_xray_mvsem_202512/current_checkpoint.pth"
metrics_file = "/root/autodl-fs/ccra_runs/iu_xray_mvsem_202512/test_metrics.json"
max_seq_length = 60

# -------------------------------
# Tokenizer
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# -------------------------------
# 数据集
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

args_dataset = type("args", (), {
    "image_dir": image_dir,
    "ann_path": ann_path,
    "max_seq_length": max_seq_length
})()

dataset = IuxrayMultiImageDataset(args_dataset, tokenizer, split="test", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# -------------------------------
# 模型
# -------------------------------
args_model = type("args", (), {
    "dataset_name": "iu_xray",
    "d_model": 512,
    "d_vf": 2048
})()
model = R2GenModel(args_model, tokenizer)
checkpoint = torch.load(checkpoint_path, map_location=device)
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# -------------------------------
# 推理 + 收集参考文本
# -------------------------------
hypotheses = {}
references = {}

with torch.no_grad():
    for batch in dataloader:
        image_ids, images, report_ids, _, _ = batch
        images = images.to(device)
        outputs, _ = model.forward_iu_xray(images, mode="sample")
        outputs = outputs.cpu().numpy()

        for img_id, out_ids, ref_ids in zip(image_ids, outputs, report_ids):
            hypo_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)
            hypotheses[img_id] = [hypo_text]
            references[img_id] = [ref_text]

# -------------------------------
# 计算指标
# -------------------------------
bleu_scorer = Bleu(4)
meteor_scorer = Meteor()
rouge_scorer = Rouge()
cider_scorer = Cider()

bleu_score, _ = bleu_scorer.compute_score(references, hypotheses)
meteor_score, _ = meteor_scorer.compute_score(references, hypotheses)
rouge_score, _ = rouge_scorer.compute_score(references, hypotheses)
cider_score, _ = cider_scorer.compute_score(references, hypotheses)

metrics = {
    "BLEU-1": bleu_score[0],
    "BLEU-2": bleu_score[1],
    "BLEU-3": bleu_score[2],
    "BLEU-4": bleu_score[3],
    "METEOR": meteor_score,
    "ROUGE_L": rouge_score,
    "CIDEr": cider_score
}

# 保存到文件
with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=4)

print("指标计算完成，结果保存在：", metrics_file)
print(metrics)
