# step_C_train_classifier.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split

# ==================== 路径配置 ====================
OUTPUT_ROOT = r"E:\SAM\23"
MASTER_LABELED = os.path.join(OUTPUT_ROOT, "02_csv_master", "masks_all_with_fix_and_label.csv")
ANALYSIS_DIR = os.path.join(OUTPUT_ROOT, "06_analysis")
# ==================================================

os.makedirs(ANALYSIS_DIR, exist_ok=True)

print("读取带标签的主表 ...")
df = pd.read_csv(MASTER_LABELED, encoding="utf-8-sig")

needed_cols = ["面积", "fixation_count", "fixation_density", "宽度", "高度", "is_important"]
for c in needed_cols:
    if c not in df.columns:
        raise ValueError(f"缺少列 '{c}'，请确认脚本 A/B 是否正确执行。")

# 删除缺失&异常
df = df.dropna(subset=needed_cols).copy()

# 特征和标签
feature_cols = ["面积", "fixation_count", "fixation_density", "宽度", "高度"]
X = df[feature_cols].values
y = df["is_important"].astype(int).values

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("训练集大小:", X_train.shape[0], "测试集大小:", X_test.shape[0])

# 随机森林分类器
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

print("开始训练模型 ...")
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
try:
    auc = roc_auc_score(y_test, y_prob)
except ValueError:
    auc = np.nan

print("\n===== 分类结果 =====")
print("Accuracy:", acc)
print("F1-score:", f1)
print("AUC:", auc)

# 保存到文本
with open(os.path.join(ANALYSIS_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {acc}\n")
    f.write(f"F1-score: {f1}\n")
    f.write(f"AUC: {auc}\n")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure()
im = plt.imshow(cm, interpolation="nearest")
plt.colorbar(im)
plt.xticks([0, 1], ["non-important", "important"])
plt.yticks([0, 1], ["non-important", "important"])
plt.xlabel("Predicted")
plt.ylabel("True")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout()
cm_path = os.path.join(ANALYSIS_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=200)
plt.close()
print("保存混淆矩阵图：", cm_path)

# ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
roc_path = os.path.join(ANALYSIS_DIR, "roc_curve.png")
plt.savefig(roc_path, dpi=200)
plt.close()
print("保存 ROC 曲线：", roc_path)

# 特征重要性
importances = clf.feature_importances_
fi_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
}).sort_values("importance", ascending=False)
fi_path = os.path.join(ANALYSIS_DIR, "feature_importances.csv")
fi_df.to_csv(fi_path, index=False, encoding="utf-8-sig")
print("保存特征重要性：", fi_path)

print("脚本 C 完成！")
