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
GT = "./data/pcw_test/test/*.npz"
MASK_PATH = "./data/pcw_test/masks/*.npz"
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

# ------------------------------------------------------------------------
# Dataset paths
# ------------------------------------------------------------------------


# ------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------


def ensure_left_to_right(lines):
    """
    Ensure each line is ordered from left (smaller x) to right (larger x).
    lines: (N, 2, 2) with format [[x0, y0], [x1, y1]]
    """
    corrected = []
    for line in lines:
        (x0, y0), (x1, y1) = line
        if x0 > x1:
            line = np.array([[x1, y1], [x0, y0]], dtype=line.dtype)
        corrected.append(line)
    return np.stack(corrected, axis=0) if len(corrected) > 0 else np.empty((0, 2, 2))


def filter_lines_with_mask(lines, scores, mask):
    """
    Keep only lines whose endpoints are inside mask==1.
    lines:  (N, 2, 2)
    scores: (N,)
    mask:   (H, W) binary array or None

    Returns (filtered_lines, filtered_scores).
    If mask is None, returns inputs unchanged.
    """
    if mask is None or lines.shape[0] == 0:
        return lines, scores

    h, w = mask.shape
    valid_lines = []
    valid_scores = []
    for line, s in zip(lines, scores):
        (a, b) = line.astype(int)
        y0, x0 = a[0], a[1]
        y1, x1 = b[0], b[1]

        if (
            0 <= y0 < h and 0 <= x0 < w and
            0 <= y1 < h and 0 <= x1 < w and
            mask[y0, x0] == 1 and mask[y1, x1] == 1
        ):
            valid_lines.append(line)
            valid_scores.append(s)

    if not valid_lines:
        return np.empty((0, 2, 2)), np.empty((0,), dtype=scores.dtype)

    return np.stack(valid_lines, axis=0), np.array(valid_scores)


def remove_duplicate_lines(lines, scores, eps=1.0):
    """
    Remove near-duplicate lines (within L2 distance < eps).
    Keeps scores in sync.
    """
    if lines is None or len(lines) == 0:
        return np.empty((0, 2, 2)), np.empty((0,), dtype=np.float32)

    unique_lines = []
    unique_scores = []
    for line, s in zip(lines, scores):
        if all(np.linalg.norm(line - ul) > eps for ul in unique_lines):
            unique_lines.append(line)
            unique_scores.append(s)

    return np.stack(unique_lines, axis=0), np.array(unique_scores)


# ------------------------------------------------------------------------
# LCNN-style masked evaluation
# ------------------------------------------------------------------------


def masked_line_score(pred_pattern, threshold=5, eps=1.0):
    """
    LCNN-style sAP with optional masking:

    - Uses lcnn.metric.msTPFP and lcnn.metric.ap
    - Computes AP globally over the entire dataset
    - Uses prediction scores to globally rank lines
    - If MASK_PATH is set and masks exist, filters predictions by mask BEFORE evaluation.
    """

    # Collect files
    pred_files = sorted(glob.glob(pred_pattern))
    gt_files = sorted(glob.glob(GT))
    mask_files = sorted(glob.glob(MASK_PATH)) if MASK_PATH is not None else []

    if not (pred_files and gt_files):
        print("Error: missing prediction or GT files.")
        return 0.0

    use_masks = MASK_PATH is not None and len(mask_files) > 0
    if MASK_PATH is not None and not mask_files:
        print("Warning: MASK_PATH is set but no mask files were found. Proceeding WITHOUT masks.")
        use_masks = False

    # Map basename -> file path
    pred_dict = {osp.splitext(osp.basename(f))[0]: f for f in pred_files}
    gt_dict = {osp.splitext(osp.basename(f))[0]: f for f in gt_files}
    mask_dict = {osp.splitext(osp.basename(f))[0]: f for f in mask_files} if use_masks else {}

    # Determine common filenames
    if use_masks:
        common = sorted(set(pred_dict.keys()) & set(gt_dict.keys()) & set(mask_dict.keys()))
    else:
        common = sorted(set(pred_dict.keys()) & set(gt_dict.keys()))

    if not common:
        print("Error: no common filenames between predictions, GT, and masks (if used).")
        return 0.0

    n_gt = 0
    all_tp, all_fp, all_scores = [], [], []

    for name in common:
        pred_path = pred_dict[name]
        gt_path = gt_dict[name]
        mask_path = mask_dict[name] if use_masks else None

        # Load prediction
        with np.load(pred_path) as fpred:
            pred_lines = fpred["lines"][:, :, :2]  # (N, 2, 2)
            pred_scores = fpred["score"]          # (N,)

        # Load ground-truth
        with np.load(gt_path) as fgt:
            gt_lines = fgt["lpos"][:, :, :2]      # (M, 2, 2)

        # Ensure consistent orientation
        pred_lines = ensure_left_to_right(pred_lines)
        gt_lines = ensure_left_to_right(gt_lines)

        # Load and apply mask if available
        mask = None
        if mask_path is not None:
            with np.load(mask_path) as fmask:
                mask = fmask["mask"]

        pred_lines, pred_scores = filter_lines_with_mask(pred_lines, pred_scores, mask)
        pred_lines, pred_scores = remove_duplicate_lines(pred_lines, pred_scores, eps)

        # Count GT lines (even if we end up with 0 preds -> affects recall denominator)
        n_gt += len(gt_lines)

        # If after masking there are no predictions, skip TP/FP but keep n_gt
        if pred_lines.shape[0] == 0:
            continue

        # Use LCNN's metric for comparability
        tp, fp = lcnn.metric.msTPFP(pred_lines, gt_lines, threshold)

        # Sanity: tp/fp length must match number of predictions
        if len(tp) != pred_lines.shape[0]:
            print(f"Warning: TP/FP length mismatch for file {name}")
            min_len = min(len(tp), pred_lines.shape[0])
            tp = tp[:min_len]
            fp = fp[:min_len]
            pred_scores = pred_scores[:min_len]

        all_tp.append(tp)
        all_fp.append(fp)
        all_scores.append(pred_scores)

    if n_gt == 0 or not all_tp:
        print("Warning: no GT lines or no valid predictions after masking.")
        return 0.0

    all_tp = np.concatenate(all_tp)
    all_fp = np.concatenate(all_fp)
    all_scores = np.concatenate(all_scores)

    # Global ranking by score (descending), as in original LCNN code
    order = np.argsort(-all_scores)
    all_tp = np.cumsum(all_tp[order]) / n_gt
    all_fp = np.cumsum(all_fp[order]) / n_gt

    return lcnn.metric.ap(all_tp, all_fp)


# ------------------------------------------------------------------------
# Optional plotting helper (not used in metric, but kept if you want visual checks)
# ------------------------------------------------------------------------


def plot_data(pred_lines_example, gt_lines_example, threshold_example, tp_example, fp_example):
    """Plots GT and predicted lines for debugging."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # 1. GT & Pred (TP in green, FP in red)
    ax = axs[0]
    ax.set_title("GT & Predicted (TP in Green, FP in Red)")
    for line in gt_lines_example:
        ax.plot([line[0][1], line[1][1]], [line[0][0], line[1][0]], 'o-')  # GT
    for i, line in enumerate(pred_lines_example):
        color = 'g' if tp_example[i] == 1 else 'r'
        ax.plot([line[0][1], line[1][1]], [line[0][0], line[1][0]], f'{color}-', linewidth=2)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 150)
    ax.invert_yaxis()

    # 2. GT only
    ax = axs[1]
    ax.set_title("GT Only")
    for line in gt_lines_example:
        ax.plot([line[0][1], line[1][1]], [line[0][0], line[1][0]], 'o-')
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 150)
    ax.invert_yaxis()

    # 3. Pred only
    ax = axs[2]
    ax.set_title("Predicted (TP in Green, FP in Red)")
    for i, line in enumerate(pred_lines_example):
        color = 'g' if tp_example[i] == 1 else 'r'
        ax.plot([line[0][1], line[1][1]], [line[0][0], line[1][0]], f'{color}-', linewidth=2)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 150)
    ax.invert_yaxis()

    plt.show()


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------

if __name__ == "__main__":
    args = docopt(__doc__)

    def work(path):
        print(f"Working on {path}")
        ms = []
        for t in [5, 10, 15]:
            print(f"\nRunning masked_line_score for threshold t={t}\n")
            score = 100.0 * masked_line_score(f"{path}/*.npz", threshold=t)
            print(f"sAP{t}: {score:.2f}")
            ms.append(score)
        return ms

    # Expand all directory patterns
    dirs = sorted(sum([glob.glob(p) for p in args["<path>"]], []))

    # Parallel map across dirs (same as original LCNN script)
    results = lcnn.utils.parmap(work, dirs)

    # Print summary per directory
    for d, msAP in zip(dirs, results):
        if msAP is None or len(msAP) != 3:
            print(f"{d}: (no valid results)")
            continue
        print(f"{d}: {msAP[0]:2.1f} {msAP[1]:2.1f} {msAP[2]:2.1f}")
