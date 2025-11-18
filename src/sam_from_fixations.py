"""
Batch SAM segmentation from eye-tracking fixations.

Author: ChatGPT (GPT-5 Pro)
Inputs:
  1) HDQ_CleanedData.csv – eye-tracking fixations with StimulusName, FixationX, FixationY.
  2) A folder containing the stimulus images (e.g., E:\SAM\All Images).
  3) A SAM checkpoint file (model weights), e.g., sam_vit_b_01ec64.pth.

Outputs (under OUTPUT_DIR):
  - 01_csv_per_image/<image_name>.csv : per-image table with [index, center_x, center_y, x_min, y_min, width, height, area, pred_iou, stability_score]
  - 02_csv_master/masks_all.csv : aggregation of all rows for all images
  - 03_json/masks_by_image.jsonl : one JSON object per image with {"image_name": ..., "masks": [{"index": i, "bbox": [x, y, w, h], "area": ..., "point_coords": [x, y]}]}
  - 04_annotated/<image_name>_annotated.png : annotated image with mask numbers, points and bounding boxes
  - 05_masks/<image_name>/<index>.png : optional binary mask files (0/255), disabled by default to save disk
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import DBSCAN

# --- SAM imports (install "segment-anything" package) ---
try:
    from segment_anything import sam_model_registry, SamPredictor
except Exception as e:
    raise RuntimeError(
        "segment-anything is not installed. Install with:\n"
        "pip install git+https://github.com/facebookresearch/segment-anything\n"
        f"Original import error: {e}"
    )

# ======================= CONFIG =======================
# ✅ 改成你的真实路径
CSV_PATH       = r"E:\SAM\数据\HDQ_CleanedData.csv"   # path to HDQ_CleanedData.csv
IMAGE_DIR      = r"E:\SAM\All Images"                # folder containing images
OUTPUT_DIR     = r"E:\SAM\12"                        # output root folder
SAM_CHECKPOINT = r"E:\SAM\sam_vit_b_01ec64.pth"      # local path to SAM checkpoint
MODEL_TYPE     = "vit_b"                             # one of: vit_h | vit_l | vit_b

# 聚类参数
CLUSTER_EPS_PX   = 35
MIN_SAMPLES      = 3
MAX_SEEDS_PER_IMAGE = 60

# SAM mask选择策略
SELECT_POLICY    = "best_iou"    # or "largest"

# 是否保存二值mask
SAVE_MASKS_TO_DISK = False

# 可识别的图片扩展名
IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

AUTO_CLIP_POINTS_TO_IMAGE = True
# ======================================================


@dataclass
class MaskInfo:
    index: int
    bbox: Tuple[int, int, int, int]           # [x_min, y_min, w, h]
    area: int
    point_coords: Tuple[float, float]         # [x, y]
    pred_iou: float | None = None
    stability_score: float | None = None


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for sub in ["01_csv_per_image", "02_csv_master", "03_json", "04_annotated", "05_masks"]:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)


def read_fixations(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["StimulusName", "FixationX", "FixationY"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Input CSV missing required column: {c}")
    df = df.dropna(subset=["StimulusName", "FixationX", "FixationY"])
    return df


def map_stimulus_to_image_path(stimulus_name: str, image_dir: str) -> str | None:
    for ext in IMG_EXTS:
        p = os.path.join(image_dir, stimulus_name + ext)
        if os.path.exists(p):
            return p
    p = os.path.join(image_dir, stimulus_name)
    if os.path.exists(p):
        return p
    return None


def dbscan_cluster(points: np.ndarray, eps: float, min_samples: int, max_keep: int | None) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if min_samples <= 1 and eps <= 0:
        centers = points.astype(np.float32)
    else:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(points)
        centers = []
        for lab in set(labels):
            if lab == -1:
                noise_pts = points[labels == -1]
                centers.extend(noise_pts.tolist())
            else:
                cluster_pts = points[labels == lab]
                cx = float(np.median(cluster_pts[:, 0]))
                cy = float(np.median(cluster_pts[:, 1]))
                centers.append([cx, cy])
        centers = np.array(centers, dtype=np.float32)
    if max_keep is not None and len(centers) > max_keep:
        centers = centers[:max_keep]
    return centers


def load_sam(model_type: str, checkpoint: str, device: str = None):
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor, device


def compute_bbox_and_area(mask: np.ndarray) -> Tuple[Tuple[int, int, int, int], int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0), 0
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    w = int(x_max - x_min + 1)
    h = int(y_max - y_min + 1)
    area = int(mask.sum())
    return (x_min, y_min, w, h), area


def draw_annotation(img: np.ndarray, masks: List[MaskInfo]) -> np.ndarray:
    out = img.copy()
    H, W = out.shape[:2]
    for m in masks:
        x, y, w, h = m.bbox
        cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), (0, 0, 255), 2)
    for m in masks:
        cx, cy = int(round(m.point_coords[0])), int(round(m.point_coords[1]))
        cv2.circle(out, (cx, cy), 4, (0, 255, 255), -1)
        text = str(m.index)
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx, ty = cx + 6, cy - 6
        tx = max(0, min(tx, W - tw - 4))
        ty = max(th + 2, min(ty, H - 2))
        cv2.rectangle(out, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (W - 1, H - 1), (0, 0, 255), 2)
    return out


def pick_mask(masks: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, float]:
    if masks is None or len(masks) == 0:
        return None, 0.0
    if SELECT_POLICY == "largest":
        areas = [m.sum() for m in masks]
        idx = int(np.argmax(areas))
    else:
        idx = int(np.argmax(scores if scores is not None else [0] * len(masks)))
    return masks[idx], float(scores[idx] if scores is not None else 0.0)


def run_sam_on_image(predictor: SamPredictor,
                     image_bgr: np.ndarray,
                     seed_points_xy: np.ndarray,
                     device: str) -> List[MaskInfo]:
    masks_info: List[MaskInfo] = []
    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    for idx, (x, y) in enumerate(seed_points_xy, start=1):
        H, W = image_bgr.shape[:2]
        if AUTO_CLIP_POINTS_TO_IMAGE:
            x = float(np.clip(x, 0, W - 1))
            y = float(np.clip(y, 0, H - 1))
        point_coords = np.array([[x, y]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        masks, ious, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=True)
        chosen_mask, best_score = pick_mask(masks, ious)
        if chosen_mask is None:
            continue
        bbox, area = compute_bbox_and_area(chosen_mask)
        mi = MaskInfo(
            index=idx,
            bbox=bbox,
            area=int(area),
            point_coords=(float(x), float(y)),
            pred_iou=float(best_score),
            stability_score=None
        )
        masks_info.append(mi)
    return masks_info


def save_per_image_csv(image_name: str, masks: List[MaskInfo]):
    rows = []
    for m in masks:
        x, y, w, h = m.bbox
        rows.append({
            "index": m.index,
            "中心点X": m.point_coords[0],
            "中心点Y": m.point_coords[1],
            "x_min": x,
            "y_min": y,
            "宽度": w,
            "高度": h,
            "面积": m.area,
            "pred_iou": m.pred_iou,
            "stability_score": m.stability_score
        })
    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUTPUT_DIR, "01_csv_per_image", f"{image_name}.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out_csv, df


def append_master_csv(image_name: str, df_image: pd.DataFrame):
    df = df_image.copy()
    df.insert(0, "ImageName", image_name)
    master_csv = os.path.join(OUTPUT_DIR, "02_csv_master", "masks_all.csv")
    header = not os.path.exists(master_csv)
    df.to_csv(master_csv, mode="a", header=header, index=False, encoding="utf-8-sig")


def save_jsonl(image_name: str, masks: List[MaskInfo]):
    obj = {
        "image_name": image_name,
        "mask_list": [{
            "index": m.index,
            "bbox": list(m.bbox),
            "area": int(m.area),
            "point_coords": [float(m.point_coords[0]), float(m.point_coords[1])]
        } for m in masks]
    }
    jsonl = os.path.join(OUTPUT_DIR, "03_json", "masks_by_image.jsonl")
    with open(jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def process_one_image(predictor: SamPredictor, image_path: str, points_xy: np.ndarray, device: str):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    masks_info = run_sam_on_image(predictor, img, points_xy, device)
    annotated = draw_annotation(img, masks_info)
    return masks_info, annotated


def main():
    ensure_dirs()
    print(f"Reading fixations from: {CSV_PATH}")
    raw = read_fixations(CSV_PATH)
    grouped = {stim: g[["FixationX", "FixationY"]].values.astype(np.float32) for stim, g in raw.groupby("StimulusName")}
    print(f"Loading SAM '{MODEL_TYPE}' from checkpoint: {SAM_CHECKPOINT}")
    predictor, device = load_sam(MODEL_TYPE, SAM_CHECKPOINT)
    pbar = tqdm(grouped.items(), desc="Images")
    for stimulus_name, pts in pbar:
        pbar.set_postfix_str(stimulus_name)
        image_path = map_stimulus_to_image_path(stimulus_name, IMAGE_DIR)
        if image_path is None:
            print(f"[WARN] Image not found for StimulusName='{stimulus_name}'. Skipping.")
            continue
        seeds = dbscan_cluster(pts, eps=CLUSTER_EPS_PX, min_samples=MIN_SAMPLES, max_keep=MAX_SEEDS_PER_IMAGE)
        if len(seeds) == 0:
            print(f"[INFO] No seed points for {stimulus_name}. Skipping.")
            continue
        try:
            masks_info, annotated = process_one_image(predictor, image_path, seeds, device=device)
        except Exception as e:
            print(f"[ERROR] Failed on {stimulus_name}: {e}")
            continue
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        out_png = os.path.join(OUTPUT_DIR, "04_annotated", f"{image_name}_annotated.png")
        cv2.imwrite(out_png, annotated)
        out_csv, df_img = save_per_image_csv(image_name, masks_info)
        append_master_csv(image_name, df_img)
        save_jsonl(image_name, masks_info)
    print("✅ Done. Results under:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
