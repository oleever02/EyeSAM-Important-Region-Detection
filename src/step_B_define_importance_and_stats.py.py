# step_B_define_importance_and_stats.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==================== 路径配置 ====================
OUTPUT_ROOT = r"E:\SAM\23"
MASTER_WITH_FIX = os.path.join(OUTPUT_ROOT, "02_csv_master", "masks_all_with_fix.csv")
OUT_MASTER_LABELED = os.path.join(OUTPUT_ROOT, "02_csv_master", "masks_all_with_fix_and_label.csv")
ANALYSIS_DIR = os.path.join(OUTPUT_ROOT, "06_analysis")
# ==================================================

os.makedirs(ANALYSIS_DIR, exist_ok=True)

print("读取 masks_all_with_fix.csv ...")
df = pd.read_csv(MASTER_WITH_FIX, encoding="utf-8-sig")

if "fixation_count" not in df.columns:
    raise ValueError("找不到列 'fixation_count'，请先运行脚本 A。")

# 确保需要的特征存在
for c in ["面积", "fixation_density"]:
    if c not in df.columns:
        raise ValueError(f"找不到列 '{c}'，请确认脚本 A 是否正确生成。")

# 1) 每张图内：按 fixation_count 排序，Top 20% 标为重要区域
df["is_important"] = 0

grouped = df.groupby("ImageName")
for name, g in grouped:
    if len(g) == 0:
        continue
    # Top 20%（至少 1 个）
    k = max(1, int(np.ceil(len(g) * 0.2)))
    g_sorted = g.sort_values("fixation_count", ascending=False)
    important_indices = g_sorted.head(k).index
    df.loc[important_indices, "is_important"] = 1

print("已根据每张图的 Top20% fixation_count 标记重要区域。")

# 2) 整体统计
overall_stats = df[["面积", "fixation_count", "fixation_density"]].describe()
overall_stats.to_csv(os.path.join(ANALYSIS_DIR, "overall_stats.csv"), encoding="utf-8-sig")

# 按重要 / 非重要分别统计
imp_stats = df[df["is_important"] == 1][["面积", "fixation_count", "fixation_density"]].describe()
nonimp_stats = df[df["is_important"] == 0][["面积", "fixation_count", "fixation_density"]].describe()

imp_stats.to_csv(os.path.join(ANALYSIS_DIR, "important_stats.csv"), encoding="utf-8-sig")
nonimp_stats.to_csv(os.path.join(ANALYSIS_DIR, "nonimportant_stats.csv"), encoding="utf-8-sig")

print("已写出统计表到 06_analysis 目录。")

# 3) 画直方图对比
def plot_hist_compare(col, bins=30):
    plt.figure()
    df[df["is_important"] == 1][col].plot(kind="hist", bins=bins, alpha=0.6, label="important", density=True)
    df[df["is_important"] == 0][col].plot(kind="hist", bins=bins, alpha=0.6, label="non-important", density=True)
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(ANALYSIS_DIR, f"hist_compare_{col}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("保存直方图：", out_path)

for col in ["面积", "fixation_count", "fixation_density"]:
    plot_hist_compare(col)

# 保存带标签的主表
df.to_csv(OUT_MASTER_LABELED, index=False, encoding="utf-8-sig")
print("写出带 is_important 标签的表：", OUT_MASTER_LABELED)
print("脚本 B 完成！")
