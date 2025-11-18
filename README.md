# EyeSAM: Important Region Detection with Eye-tracking and SAM

本仓库实现了一个基于 **眼动数据 + Segment Anything Model (SAM)** 的“重要区域”分析小项目。

给定：
- 一组图像
- 对应的眼动注视点（FixationX, FixationY）

我们：
1. 使用 SAM 将图像分割成多个 mask 区域；
2. 统计每个 mask 内的注视点数量（fixation_count），以及单位面积注视密度（fixation_density）；
3. 在每张图像中，将 fixation_count 排名前 20% 的 mask 定义为 **重要区域 (important)**，其余为 **非重要区域 (non-important)**；
4. 利用简单的几何 + 眼动统计特征训练分类模型，预测某个区域是否为重要区域。

这是一个结合 CV 与眼动数据的完整小 pipeline，适合作为课程项目 / 个人练习 demo。

---

## 1. 环境安装 (Installation)

```bash
git clone https://github.com/oleever02/EyeSAM-Important-Region-Detection.git
cd EyeSAM-Important-Region-Detection

pip install -r requirements.txt
