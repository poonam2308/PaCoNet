#!/usr/bin/env python3
"""Evaluate sAP5, sAP10, sAP15 (from LCNN)
Usage:
    eval-sAP.py <path>...
    eval-sAP.py (-h | --help )

Examples:
    python eval-sAP.py logs/*/npz/000*

Arguments:
    <path>                           One or more directories from train.py

Options:
   -h --help                         Show this screen.
"""

import os
import sys
import glob
import os.path as osp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt
import random
import torch
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))  # 2 levels up from this file
sys.path.insert(0, project_root)
import src.dhlp.lcnn.utils
import src.dhlp.lcnn.metric
from src.dhlp import lcnn

# color
# GT = "./data/pcw_test/test/*.npz"
# MASK_PATH = "./data/pcw_test/masks/*.npz"
#
# # cluster
# GT = "./data/pcw_test_cls/test/*.npz"
# MASK_PATH = "./data/pcw_test_cls/masks/*.npz"
#
# # color no unet
# GT = "./data/pcw_ntest/test/*.npz"
# MASK_PATH = "./data/pcw_ntest/masks/*.npz"
#
# # cluster no unet
# GT = "./data/pcw_ntest_cls/test/*.npz"
# MASK_PATH = "./data/pcw_ntest_cls/masks/*.npz"



# ---------- EDIT THESE PATHS ----------
# eval_sAP_with_mask_normmatch.py
# Matches pred/gt/mask by normalized filename id (handles _label/_mask/_pred suffixes).
# No argument parser: edit paths below and run:
#   python eval_sAP_with_mask_normmatch.py

import os
import glob
import numpy as np

# ---------- EDIT THESE ----------
PRED_DIR   = "./outputs/results/c"             # folder containing subset of prediction .npz
GT_GLOB    = "./data/pcw_test/test/*.npz"      # glob for ALL GT .npz
MASK_GLOB  = "./data/pcw_test/masks/*.npz"     # glob for ALL mask .npz (must contain key "mask")
# -------------------------------

MASK_TYPE = "npz"  # this script expects npz masks if MASK_GLOB is npz

# ---------- FILTER SETTINGS ----------
KEEP_RATIO = 0.50
SAMPLES    = 25
THRESHOLDS = [5, 10, 15]   # pixels in SAME scale as your labels/preds (128-scale)
# ------------------------------------

# If your naming differs, add/remove suffixes here.
# These will be stripped repeatedly from the end of filenames until stable.
STRIP_SUFFIXES = [
    "_label", "_labels",
    "_mask", "_masks",
    "_pred", "_preds", "_prediction", "_predictions",
    "_result", "_results",
    "_output", "_outputs",
]


def line_angle_deg(line):
    """
    line: (2,2) in (y,x)
    returns angle in degrees in [0, 180)
    """
    dy, dx = line[1] - line[0]
    return np.degrees(np.arctan2(dy, dx)) % 180

def perp_distance(line1, line2):
    """
    Mean perpendicular distance between endpoints of line2 to line1.
    Assumes near-parallel lines.
    """
    p0, p1 = line1
    v = p1 - p0
    v = v / (np.linalg.norm(v) + 1e-8)

    # normal vector
    n = np.array([-v[1], v[0]])

    d0 = abs(np.dot(line2[0] - p0, n))
    d1 = abs(np.dot(line2[1] - p0, n))
    return float(0.5 * (d0 + d1))

def line_nms_parallel(lines, scores,
                      angle_thresh=8.0,     # degrees
                      perp_dist_thresh=3.0  # pixels (128-scale)
                     ):
    """
    NMS specialized for many parallel lines (your dataset).
    """
    if len(lines) == 0:
        return lines, scores

    angles = np.array([line_angle_deg(l) for l in lines])
    order = np.argsort(-scores)
    keep = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        rest = order[1:]
        survivors = []

        for j in rest:
            j = int(j)

            # 1️⃣ angle check
            da = abs(angles[i] - angles[j])
            da = min(da, 180 - da)
            if da > angle_thresh:
                survivors.append(j)
                continue

            # 2️⃣ perpendicular distance check
            d = perp_distance(lines[i], lines[j])
            if d > perp_dist_thresh:
                survivors.append(j)
                continue

            # else → suppress (parallel + too close)

        order = np.array(survivors, dtype=int)

    return lines[keep], scores[keep]


def _line_vec(line):
    """line: (2,2) in (y,x). returns direction vector in (dy,dx)."""
    v = line[1] - line[0]
    n = np.linalg.norm(v) + 1e-8
    return v / n

def _angle_diff_deg(line1, line2):
    """
    Smallest angle between two undirected line segments, in degrees.
    Treats reversed direction as same (uses abs(cos)).
    """
    v1 = _line_vec(line1)
    v2 = _line_vec(line2)
    cos = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    cos = abs(cos)  # undirected: 0° == 180°
    return float(np.degrees(np.arccos(cos)))

def _endpoint_distance(line1, line2):
    """
    Order-invariant endpoint distance:
    min( ||a0-b0||+||a1-b1|| , ||a0-b1||+||a1-b0|| )
    """
    d1 = np.linalg.norm(line1[0] - line2[0]) + np.linalg.norm(line1[1] - line2[1])
    d2 = np.linalg.norm(line1[0] - line2[1]) + np.linalg.norm(line1[1] - line2[0])
    return float(min(d1, d2))

def line_nms_angle_aware(lines, scores, dist_thresh=5.0, angle_thresh_deg=15.0):
    """
    lines:  (N,2,2) in (y,x)
    scores: (N,)
    Keeps high-score lines and suppresses lines that are:
      - within dist_thresh (in label pixels, i.e., 128-scale)
      - AND within angle_thresh_deg

    Returns: (lines_kept, scores_kept)
    """
    if len(lines) == 0:
        return lines, scores

    order = np.argsort(-scores)  # high -> low
    keep = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        rest = order[1:]
        if rest.size == 0:
            break

        survivors = []
        for j in rest:
            j = int(j)
            d = _endpoint_distance(lines[i], lines[j])
            if d >= dist_thresh:
                survivors.append(j)
                continue

            a = _angle_diff_deg(lines[i], lines[j])
            if a >= angle_thresh_deg:
                survivors.append(j)
                continue

            # else: suppress j (too close AND too parallel)

        order = np.array(survivors, dtype=int)

    return lines[keep], scores[keep]

def line_distance(l1, l2):
    """
    l1, l2: (2,2) arrays in (y,x)
    returns min endpoint distance considering both directions
    """
    d1 = np.linalg.norm(l1[0] - l2[0]) + np.linalg.norm(l1[1] - l2[1])
    d2 = np.linalg.norm(l1[0] - l2[1]) + np.linalg.norm(l1[1] - l2[0])
    return min(d1, d2)


def line_nms(lines, scores, dist_thresh=5.0):
    """
    lines:  (N,2,2) in (y,x)
    scores: (N,)
    dist_thresh: distance threshold in SAME scale (128-scale)

    returns filtered (lines, scores)
    """
    if len(lines) == 0:
        return lines, scores

    order = np.argsort(-scores)  # high → low confidence
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        rest = order[1:]
        suppress = []

        for j in rest:
            if line_distance(lines[i], lines[j]) < dist_thresh:
                suppress.append(j)

        order = np.array([j for j in rest if j not in suppress], dtype=int)

    return lines[keep], scores[keep]

def norm_id(path: str) -> str:
    """Normalize a file path to a stable sample id (strip extension + known suffixes)."""
    base = os.path.splitext(os.path.basename(path))[0]
    changed = True
    while changed:
        changed = False
        for suf in STRIP_SUFFIXES:
            if base.endswith(suf):
                base = base[: -len(suf)]
                changed = True
    return base


def build_id_map(file_list):
    """Map normalized id -> full path. Keeps the first seen file per id."""
    m = {}
    for p in file_list:
        k = norm_id(p)
        if k not in m:
            m[k] = p
    return m


def load_mask(mask_path: str) -> np.ndarray:
    """Load mask as binary (H,W) array with values {0,1}."""
    ext = os.path.splitext(mask_path)[1].lower()

    if ext == ".npz":
        with np.load(mask_path) as f:
            if "mask" not in f:
                raise KeyError(f"'mask' key not found in {mask_path}. Keys={list(f.keys())}")
            m = f["mask"]
    else:
        if cv2 is None:
            raise RuntimeError("cv2 not available to read PNG masks. Install opencv-python or use npz masks.")
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"Failed to read mask image: {mask_path}")

    if m.ndim != 2:
        m = m.squeeze()
    return (m > 0).astype(np.uint8)


def line_inside_ratio(mask: np.ndarray, line: np.ndarray, samples: int) -> float:
    """
    mask: (H,W) binary
    line: (2,2) endpoints in (y,x)
    returns fraction of sampled points inside mask
    """
    H, W = mask.shape
    (y0, x0), (y1, x1) = line

    ts = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    ys = y0 + (y1 - y0) * ts
    xs = x0 + (x1 - x0) * ts

    ys = np.clip(np.rint(ys).astype(np.int32), 0, H - 1)
    xs = np.clip(np.rint(xs).astype(np.int32), 0, W - 1)
    return float(mask[ys, xs].mean())


def filter_pred_lines(lines: np.ndarray, scores: np.ndarray, mask: np.ndarray,
                      keep_ratio: float, samples: int):
    keep = np.zeros((len(lines),), dtype=bool)
    for i in range(len(lines)):
        keep[i] = (line_inside_ratio(mask, lines[i], samples) >= keep_ratio)
    return lines[keep], scores[keep]


def compute_sap(pred_files, gt_map, mask_map, threshold: int) -> float:
    n_gt = 0
    tp_all, fp_all, sc_all = [], [], []

    used = 0
    skipped = 0

    for pred_path in pred_files:
        k = norm_id(pred_path)
        gt_path = gt_map.get(k, None)
        mask_path = mask_map.get(k, None)

        if gt_path is None or mask_path is None:
            skipped += 1
            continue

        used += 1

        with np.load(pred_path) as fp:
            pred_lines = fp["lines"][:, :, :2].astype(np.float32)
            pred_scores = fp["score"].astype(np.float32)

        with np.load(gt_path) as fg:
            gt_lines = fg["lpos"][:, :, :2].astype(np.float32)

        mask = load_mask(mask_path)

        pred_lines_f, pred_scores_f = filter_pred_lines(
            pred_lines, pred_scores, mask,
            keep_ratio=KEEP_RATIO, samples=SAMPLES
        )

        # # APPLY NMS HERE
        # pred_lines_f, pred_scores_f = line_nms(
        #     pred_lines_f,
        #     pred_scores_f,
        #     dist_thresh=5.0
        # )
        #
        pred_lines_f, pred_scores_f = line_nms_angle_aware(
            pred_lines_f, pred_scores_f,
            dist_thresh=5.0,
            angle_thresh_deg=15.0
        )

        # pred_lines, pred_scores = line_nms_parallel(
        #     pred_lines,
        #     pred_scores,
        #     angle_thresh=6.0,
        #     perp_dist_thresh=3.0
        # )

        n_gt += len(gt_lines)

        # Keep the same duplicate-early-stop behavior as your original eval script
        for i in range(len(pred_lines_f)):
            if i > 0 and (pred_lines_f[i] == pred_lines_f[0]).all():
                pred_lines_f = pred_lines_f[:i]
                pred_scores_f = pred_scores_f[:i]
                break

        tp, fp = lcnn.metric.msTPFP(pred_lines_f, gt_lines, threshold)
        tp_all.append(tp)
        fp_all.append(fp)
        sc_all.append(pred_scores_f)

    print(f"Threshold {threshold}: used {used} preds, skipped {skipped} (missing GT/mask match)")

    if n_gt == 0 or len(tp_all) == 0:
        return 0.0

    tp_all = np.concatenate(tp_all)
    fp_all = np.concatenate(fp_all)
    sc_all = np.concatenate(sc_all)

    order = np.argsort(-sc_all)
    tp_c = np.cumsum(tp_all[order]) / n_gt
    fp_c = np.cumsum(fp_all[order]) / n_gt
    return float(lcnn.metric.ap(tp_c, fp_c))


def main():
    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*.npz")))
    gt_files = sorted(glob.glob(GT_GLOB))
    mask_files = sorted(glob.glob(MASK_GLOB))

    print(f"Found {len(pred_files)} prediction files in: {PRED_DIR}")
    print(f"Found {len(gt_files)} GT files from: {GT_GLOB}")
    print(f"Found {len(mask_files)} mask files from: {MASK_GLOB}")

    if len(pred_files) == 0:
        print("No preds found. Check PRED_DIR.")
        return

    gt_map = build_id_map(gt_files)
    mask_map = build_id_map(mask_files)

    # Debug: check matching coverage
    pred_ids = [norm_id(p) for p in pred_files]
    matched = [k for k in pred_ids if (k in gt_map and k in mask_map)]
    print(f"Matched preds to GT+mask by normalized id: {len(matched)}/{len(pred_files)}")

    if len(matched) == 0:
        print("\n[DEBUG] Example IDs (first 5):")
        print("  pred ids :", pred_ids[:5])
        print("  gt ids   :", list(gt_map.keys())[:5])
        print("  mask ids :", list(mask_map.keys())[:5])
        print("\nIf these look similar but not equal, add the right suffix in STRIP_SUFFIXES.")
        return

    for t in THRESHOLDS:
        ap = compute_sap(pred_files, gt_map, mask_map, threshold=t)
        print(f"sAP@{t} (masked): {ap * 100.0:.2f}")


if __name__ == "__main__":
    main()
























#
# # ------------------------------------------------------------------------
# # Dataset paths
# # ------------------------------------------------------------------------
#
#
# # ------------------------------------------------------------------------
# # Utility functions
# # ------------------------------------------------------------------------
#
#
# def ensure_left_to_right(lines):
#     """
#     Ensure each line is ordered from left (smaller x) to right (larger x).
#     lines: (N, 2, 2) with format [[x0, y0], [x1, y1]]
#     """
#     corrected = []
#     for line in lines:
#         (x0, y0), (x1, y1) = line
#         if x0 > x1:
#             line = np.array([[x1, y1], [x0, y0]], dtype=line.dtype)
#         corrected.append(line)
#     return np.stack(corrected, axis=0) if len(corrected) > 0 else np.empty((0, 2, 2))
#
#
# def filter_lines_with_mask(lines, scores, mask):
#     """
#     Keep only lines whose endpoints are inside mask==1.
#     lines:  (N, 2, 2)
#     scores: (N,)
#     mask:   (H, W) binary array or None
#
#     Returns (filtered_lines, filtered_scores).
#     If mask is None, returns inputs unchanged.
#     """
#     if mask is None or lines.shape[0] == 0:
#         return lines, scores
#
#     h, w = mask.shape
#     valid_lines = []
#     valid_scores = []
#     for line, s in zip(lines, scores):
#         (a, b) = line.astype(int)
#         y0, x0 = a[0], a[1]
#         y1, x1 = b[0], b[1]
#
#         if (
#             0 <= y0 < h and 0 <= x0 < w and
#             0 <= y1 < h and 0 <= x1 < w and
#             mask[y0, x0] == 1 and mask[y1, x1] == 1
#         ):
#             valid_lines.append(line)
#             valid_scores.append(s)
#
#     if not valid_lines:
#         return np.empty((0, 2, 2)), np.empty((0,), dtype=scores.dtype)
#
#     return np.stack(valid_lines, axis=0), np.array(valid_scores)
#
#
# def remove_duplicate_lines(lines, scores, eps=1.0):
#     """
#     Remove near-duplicate lines (within L2 distance < eps).
#     Keeps scores in sync.
#     """
#     if lines is None or len(lines) == 0:
#         return np.empty((0, 2, 2)), np.empty((0,), dtype=np.float32)
#
#     unique_lines = []
#     unique_scores = []
#     for line, s in zip(lines, scores):
#         if all(np.linalg.norm(line - ul) > eps for ul in unique_lines):
#             unique_lines.append(line)
#             unique_scores.append(s)
#
#     return np.stack(unique_lines, axis=0), np.array(unique_scores)
#
#
# # ------------------------------------------------------------------------
# # LCNN-style masked evaluation
# # ------------------------------------------------------------------------
#
#
# def masked_line_score(pred_pattern, threshold=5, eps=1.0):
#     """
#     LCNN-style sAP with optional masking:
#
#     - Uses lcnn.metric.msTPFP and lcnn.metric.ap
#     - Computes AP globally over the entire dataset
#     - Uses prediction scores to globally rank lines
#     - If MASK_PATH is set and masks exist, filters predictions by mask BEFORE evaluation.
#     """
#
#     # Collect files
#     pred_files = sorted(glob.glob(pred_pattern))
#     gt_files = sorted(glob.glob(GT))
#     mask_files = sorted(glob.glob(MASK_PATH)) if MASK_PATH is not None else []
#
#     if not (pred_files and gt_files):
#         print("Error: missing prediction or GT files.")
#         return 0.0
#
#     use_masks = MASK_PATH is not None and len(mask_files) > 0
#     if MASK_PATH is not None and not mask_files:
#         print("Warning: MASK_PATH is set but no mask files were found. Proceeding WITHOUT masks.")
#         use_masks = False
#
#     # Map basename -> file path
#     pred_dict = {osp.splitext(osp.basename(f))[0]: f for f in pred_files}
#     gt_dict = {osp.splitext(osp.basename(f))[0]: f for f in gt_files}
#     mask_dict = {osp.splitext(osp.basename(f))[0]: f for f in mask_files} if use_masks else {}
#
#     # Determine common filenames
#     if use_masks:
#         common = sorted(set(pred_dict.keys()) & set(gt_dict.keys()) & set(mask_dict.keys()))
#     else:
#         common = sorted(set(pred_dict.keys()) & set(gt_dict.keys()))
#
#     if not common:
#         print("Error: no common filenames between predictions, GT, and masks (if used).")
#         return 0.0
#
#     n_gt = 0
#     all_tp, all_fp, all_scores = [], [], []
#
#     for name in common:
#         pred_path = pred_dict[name]
#         gt_path = gt_dict[name]
#         mask_path = mask_dict[name] if use_masks else None
#
#         # Load prediction
#         with np.load(pred_path) as fpred:
#             pred_lines = fpred["lines"][:, :, :2]  # (N, 2, 2)
#             pred_scores = fpred["score"]          # (N,)
#
#         # Load ground-truth
#         with np.load(gt_path) as fgt:
#             gt_lines = fgt["lpos"][:, :, :2]      # (M, 2, 2)
#
#         # Ensure consistent orientation
#         pred_lines = ensure_left_to_right(pred_lines)
#         gt_lines = ensure_left_to_right(gt_lines)
#
#         # Load and apply mask if available
#         mask = None
#         if mask_path is not None:
#             with np.load(mask_path) as fmask:
#                 mask = fmask["mask"]
#
#         pred_lines, pred_scores = filter_lines_with_mask(pred_lines, pred_scores, mask)
#         pred_lines, pred_scores = remove_duplicate_lines(pred_lines, pred_scores, eps)
#
#         # Count GT lines (even if we end up with 0 preds -> affects recall denominator)
#         n_gt += len(gt_lines)
#
#         # If after masking there are no predictions, skip TP/FP but keep n_gt
#         if pred_lines.shape[0] == 0:
#             continue
#
#         # Use LCNN's metric for comparability
#         tp, fp = lcnn.metric.msTPFP(pred_lines, gt_lines, threshold)
#
#         # Sanity: tp/fp length must match number of predictions
#         if len(tp) != pred_lines.shape[0]:
#             print(f"Warning: TP/FP length mismatch for file {name}")
#             min_len = min(len(tp), pred_lines.shape[0])
#             tp = tp[:min_len]
#             fp = fp[:min_len]
#             pred_scores = pred_scores[:min_len]
#
#         all_tp.append(tp)
#         all_fp.append(fp)
#         all_scores.append(pred_scores)
#
#     if n_gt == 0 or not all_tp:
#         print("Warning: no GT lines or no valid predictions after masking.")
#         return 0.0
#
#     all_tp = np.concatenate(all_tp)
#     all_fp = np.concatenate(all_fp)
#     all_scores = np.concatenate(all_scores)
#
#     # Global ranking by score (descending), as in original LCNN code
#     order = np.argsort(-all_scores)
#     all_tp = np.cumsum(all_tp[order]) / n_gt
#     all_fp = np.cumsum(all_fp[order]) / n_gt
#
#     return lcnn.metric.ap(all_tp, all_fp)
#
#
# # ------------------------------------------------------------------------
# # Optional plotting helper (not used in metric, but kept if you want visual checks)
# # ------------------------------------------------------------------------
#
#
# def plot_data(pred_lines_example, gt_lines_example, threshold_example, tp_example, fp_example):
#     """Plots GT and predicted lines for debugging."""
#     fig, axs = plt.subplots(1, 3, figsize=(18, 6))
#
#     # 1. GT & Pred (TP in green, FP in red)
#     ax = axs[0]
#     ax.set_title("GT & Predicted (TP in Green, FP in Red)")
#     for line in gt_lines_example:
#         ax.plot([line[0][1], line[1][1]], [line[0][0], line[1][0]], 'o-')  # GT
#     for i, line in enumerate(pred_lines_example):
#         color = 'g' if tp_example[i] == 1 else 'r'
#         ax.plot([line[0][1], line[1][1]], [line[0][0], line[1][0]], f'{color}-', linewidth=2)
#     ax.set_xlim(0, 150)
#     ax.set_ylim(0, 150)
#     ax.invert_yaxis()
#
#     # 2. GT only
#     ax = axs[1]
#     ax.set_title("GT Only")
#     for line in gt_lines_example:
#         ax.plot([line[0][1], line[1][1]], [line[0][0], line[1][0]], 'o-')
#     ax.set_xlim(0, 150)
#     ax.set_ylim(0, 150)
#     ax.invert_yaxis()
#
#     # 3. Pred only
#     ax = axs[2]
#     ax.set_title("Predicted (TP in Green, FP in Red)")
#     for i, line in enumerate(pred_lines_example):
#         color = 'g' if tp_example[i] == 1 else 'r'
#         ax.plot([line[0][1], line[1][1]], [line[0][0], line[1][0]], f'{color}-', linewidth=2)
#     ax.set_xlim(0, 150)
#     ax.set_ylim(0, 150)
#     ax.invert_yaxis()
#
#     plt.show()
#
#
# # ------------------------------------------------------------------------
# # Main
# # ------------------------------------------------------------------------
#
# if __name__ == "__main__":
#     args = docopt(__doc__)
#
#     def work(path):
#         print(f"Working on {path}")
#         ms = []
#         for t in [5, 10, 15]:
#             print(f"\nRunning masked_line_score for threshold t={t}\n")
#             score = 100.0 * masked_line_score(f"{path}/*.npz", threshold=t)
#             print(f"sAP{t}: {score:.2f}")
#             ms.append(score)
#         return ms
#
#     # Expand all directory patterns
#     dirs = sorted(sum([glob.glob(p) for p in args["<path>"]], []))
#
#     # Parallel map across dirs (same as original LCNN script)
#     results = lcnn.utils.parmap(work, dirs)
#
#     # Print summary per directory
#     for d, msAP in zip(dirs, results):
#         if msAP is None or len(msAP) != 3:
#             print(f"{d}: (no valid results)")
#             continue
#         print(f"{d}: {msAP[0]:2.1f} {msAP[1]:2.1f} {msAP[2]:2.1f}")
