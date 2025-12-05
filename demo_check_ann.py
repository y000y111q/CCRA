import json

ann_path = "data/iu_xray/annotation.json"   # 按你实际路径改

with open(ann_path, "r", encoding="utf-8") as f:
    ann = json.load(f)

print("keys:", ann.keys())
print("train size:", len(ann["train"]))
print("val size:", len(ann["val"]))
print("test size:", len(ann["test"]))

print("\n=== 示例一条 train ===")
sample = ann["train"][0]
print("id:", sample.get("id"))
print("image_path:", sample.get("image_path"))
print("report:", sample.get("report"))

import json
import re

ann_path = "data/iu_xray/annotation.json"

with open(ann_path, "r", encoding="utf-8") as f:
    ann = json.load(f)

all_reports = []
for split in ["train", "val", "test"]:
    for item in ann[split]:
        all_reports.append((split, item["id"], item["report"]))

# 看看有没有含有典型“诊断段”句子的例子，比如 Normal chest x-ray.
cnt_impression_like = 0
for split, id_, rep in all_reports:
    if re.search(r"Normal chest", rep, re.IGNORECASE):
        cnt_impression_like += 1
        print("example:", split, id_, rep)
        if cnt_impression_like >= 3:
            break

print("包含 'Normal chest' 的报告数量:", cnt_impression_like)

