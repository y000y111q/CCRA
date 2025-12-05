import pandas as pd
import matplotlib.pyplot as plt

# ======== 需要你自己改的三个名字 ========
CSV_NAME = "results_mvsem_ccra.csv"          # 这次要画的那个实验的 csv 文件名
OUTPUT_FIG = "metrics_mvsem_ccra.png"        # 输出的图片名
TITLE = "MVSEM+CCRA on IU X-Ray (test)"      # 图标题，想怎么写都行
# =====================================

# 1. 读入该实验的 CSV
df = pd.read_csv(CSV_NAME)

# 2. 只看 test 的行（以后如果有 val/train 也不会混进去）
test_df = df[df["split"] == "test"]

# 取最后一行，假设就是该实验最终结果
row = test_df.iloc[-1]

# 3. 想展示哪些指标
metric_names = ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4", "ROUGE_L", "CIDEr"]
metric_names = [m for m in metric_names if m in row.index]  # 防止有的没算

values = [row[m] for m in metric_names]

# 4. 在终端打印一个 Markdown 表格（可以粘到论文草稿/笔记里）
print("| Metric | Value |")
print("|--------|-------|")
for m, v in zip(metric_names, values):
    print(f"| {m} | {v:.4f} |")

# 5. 画柱状图并保存
plt.figure()
plt.bar(metric_names, values)
plt.ylabel("Score")
plt.title(TITLE)
plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300)
print(f"\nSaved figure to {OUTPUT_FIG}")

