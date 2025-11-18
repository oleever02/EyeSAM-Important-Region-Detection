# step_A_add_fixation_counts.py
import os
import pandas as pd
import numpy as np

# ==================== 路径配置 ====================
MASK_MASTER = r"E:\SAM\12\02_csv_master\masks_all.csv"       # 你的 masks_all_.csv
FIX_CSV     = r"E:\SAM\数据\HDQ_CleanedData.csv"    # 眼动原始数据
OUTPUT_ROOT = r"E:\SAM\23"                     # 输出根目录（和之前一样）
OUT_MASTER_DIR = os.path.join(OUTPUT_ROOT, "02_csv_master")
OUT_MASTER = os.path.join(OUT_MASTER_DIR, "masks_all_with_fix.csv")
# ==================================================

os.makedirs(OUT_MASTER_DIR, exist_ok=True)

def norm_name(x: str) -> str:
    """去掉路径和扩展名，保证 StimulusName 和 ImageName 能对应上"""
    x = str(x)
    base = os.path.basename(x)
    stem, _ = os.path.splitext(base)
    return stem

print("读取 masks_all_.csv ...")
masks = pd.read_csv(MASK_MASTER, encoding="utf-8-sig")
print("读取 HDQ_CleanedData.csv ...")
fix = pd.read_csv(FIX_CSV, encoding="utf-8-sig")

# 规范图像名
if "ImageName" not in masks.columns:
    raise ValueError("masks_all_.csv 里找不到 'ImageName' 列，请确认列名。")

masks["ImageNameNorm"] = masks["ImageName"].apply(norm_name)

if "StimulusName" not in fix.columns or "FixationX" not in fix.columns or "FixationY" not in fix.columns:
    raise ValueError("HDQ_CleanedData.csv 需要包含 'StimulusName', 'FixationX', 'FixationY' 三列。")

fix["ImageNameNorm"] = fix["StimulusName"].apply(norm_name)

# 预先加上列
masks["fixation_count"] = 0

# 检查列名（bbox/面积）
required_cols = ["x_min", "y_min", "宽度", "高度", "面积"]
for c in required_cols:
    if c not in masks.columns:
        raise ValueError(f"masks_all_.csv 里找不到列 '{c}'，请检查列名。")

# 确保数值型
for c in ["x_min", "y_min", "宽度", "高度"]:
    masks[c] = pd.to_numeric(masks[c], errors="coerce").fillna(0).astype(float)
masks["面积"] = pd.to_numeric(masks["面积"], errors="coerce").fillna(0).astype(float)

print("开始按图像统计 fixation_count ...")
grouped_masks = masks.groupby("ImageNameNorm")
grouped_fix = fix.groupby("ImageNameNorm")

from tqdm import tqdm
for name, g_masks in tqdm(grouped_masks, desc="Images"):
    # 找到该图像对应的 fixations
    if name not in grouped_fix.indices:
        # 没有任何注视点，全部为 0
        continue
    g_fix = grouped_fix.get_group(name)
    fx = g_fix["FixationX"].values
    fy = g_fix["FixationY"].values

    # 对这一张图的每个 mask 统计
    idxs = g_masks.index
    for idx in idxs:
        row = masks.loc[idx]
        x_min = row["x_min"]
        y_min = row["y_min"]
        w = row["宽度"]
        h = row["高度"]

        # [x_min, x_min+w), [y_min, y_min+h)
        in_x = (fx >= x_min) & (fx < x_min + w)
        in_y = (fy >= y_min) & (fy < y_min + h)
        count = int(np.sum(in_x & in_y))
        masks.at[idx, "fixation_count"] = count

# fixation_density = count / area（防止除 0）
masks["fixation_density"] = masks.apply(
    lambda r: r["fixation_count"] / r["面积"] if r["面积"] > 0 else 0,
    axis=1
)

print("写出带有 fixation_count 的汇总表：", OUT_MASTER)
masks.to_csv(OUT_MASTER, index=False, encoding="utf-8-sig")
print("完成！")
