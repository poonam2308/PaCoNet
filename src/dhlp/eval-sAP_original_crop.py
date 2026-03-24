#!/usr/bin/env python3
"""Evaluate sAP5, sAP10, sAP15 (LCNN-style), but MERGE multiple cat files that belong
to the same crop into a single crop-level evaluation item.

Usage:
    eval-sAP_merge_crops.py <path>...
    eval-sAP_merge_crops.py (-h | --help)

Examples:
    python eval-sAP_merge_crops.py logs/*/npz/000*

Arguments:
    <path>    One or more directories containing prediction npz files.

Options:
   -h --help  Show this screen.
"""


# this is similar to merging the input from all cat, into one image and making it alike with crop image
import os
import glob
import sys
import numpy as np
from collections import defaultdict
from docopt import docopt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

import src.dhlp.lcnn.utils
import src.dhlp.lcnn.metric
from src.dhlp import lcnn
# poonam added 10/12/2025
# color
# GT = "./data/pcw_test/test/*.npz"

# # cluster
# GT = "./data/pcw_test_cls/test/*.npz"

# # color no unet
# GT = "./data/pcw_ntest/test/*.npz"

# # cluster no unet
GT = "./data/pcw_ntest_cls/test/*.npz"




def crop_key_from_npz(path: str) -> str:
    """
    Works for both pred and GT filenames like:
      image_1000_crop_1_0VSK_0.npz
      image_1000_crop_1_0VSK_0_label.npz
    We want crop key:
      image_1000_crop_1
    """
    base = os.path.basename(path)
    base = base.replace("_label.npz", "").replace(".npz", "")
    parts = base.split("_")
    # expected: image, 1000, crop, 1, <cat>, <frame>
    if len(parts) >= 4:
        return "_".join(parts[:4])
    return base


def dedupe_lines_simple(lines: np.ndarray, tol: float = 1.0) -> np.ndarray:
    """
    Deduplicate merged GT lines (or preds if you want) using a simple endpoint-quantization rule.
    lines: [N,2,2] (y,x)
    tol:   quantization bin size in pixels/coords (heatmap coords)
    """
    if lines.size == 0:
        return lines
    q = np.round(lines / tol).astype(np.int32)

    # canonical ordering of endpoints so AB == BA
    a = q[:, 0, :]
    b = q[:, 1, :]
    swap = (a[:, 0] > b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] > b[:, 1]))
    q2 = q.copy()
    q2[swap, 0, :] = b[swap]
    q2[swap, 1, :] = a[swap]

    # unique rows
    key = q2.reshape(q2.shape[0], -1)
    _, idx = np.unique(key, axis=0, return_index=True)
    return lines[np.sort(idx)]


def line_score_merged(pred_glob: str, threshold: float = 5.0) -> float:
    preds = sorted(glob.glob(pred_glob))
    gts = sorted(glob.glob(GT))

    # group pred files by crop
    pred_lines_by_crop = defaultdict(list)
    pred_scores_by_crop = defaultdict(list)

    for pred_name in preds:
        with np.load(pred_name) as fpred:
            lines = fpred["lines"][:, :, :2]
            scores = fpred["score"]

        # strip repeated padding rows (your original behavior)
        for i in range(len(lines)):
            if i > 0 and (lines[i] == lines[0]).all():
                lines = lines[:i]
                scores = scores[:i]
                break

        ck = crop_key_from_npz(pred_name)
        pred_lines_by_crop[ck].append(lines)
        pred_scores_by_crop[ck].append(scores)

    # group GT files by crop
    gt_lines_by_crop = defaultdict(list)
    for gt_name in gts:
        with np.load(gt_name) as fgt:
            gt_line = fgt["lpos"][:, :, :2]
        ck = crop_key_from_npz(gt_name)
        gt_lines_by_crop[ck].append(gt_line)

    # evaluate per crop
    n_gt = 0
    all_tp, all_fp, all_scores = [], [], []

    crops = sorted(set(gt_lines_by_crop.keys()) & set(pred_lines_by_crop.keys()))

    for ck in crops:
        # merge preds
        plines = np.concatenate(pred_lines_by_crop[ck], axis=0) if pred_lines_by_crop[ck] else np.zeros((0, 2, 2))
        pscores = np.concatenate(pred_scores_by_crop[ck], axis=0) if pred_scores_by_crop[ck] else np.zeros((0,))

        # merge GT
        glines = np.concatenate(gt_lines_by_crop[ck], axis=0) if gt_lines_by_crop[ck] else np.zeros((0, 2, 2))
        # dedupe merged GT to avoid counting same GT multiple times
        glines = dedupe_lines_simple(glines, tol=1.0)

        n_gt += len(glines)

        # compute TP/FP at this threshold for this crop
        tp, fp = lcnn.metric.msTPFP(plines, glines, threshold)
        all_tp.append(tp)
        all_fp.append(fp)
        all_scores.append(pscores)

    if n_gt == 0:
        return 0.0

    all_tp = np.concatenate(all_tp) if len(all_tp) else np.zeros((0,))
    all_fp = np.concatenate(all_fp) if len(all_fp) else np.zeros((0,))
    all_scores = np.concatenate(all_scores) if len(all_scores) else np.zeros((0,))

    # sort by score descending
    idx = np.argsort(-all_scores)
    tp_curve = np.cumsum(all_tp[idx]) / n_gt
    fp_curve = np.cumsum(all_fp[idx]) / n_gt

    return lcnn.metric.ap(tp_curve, fp_curve)


if __name__ == "__main__":
    args = docopt(__doc__)

    def work(path):
        print(f"Working on {path}")
        return [100 * line_score_merged(f"{path}/*.npz", t) for t in [5, 10, 15]]

    dirs = sorted(sum([glob.glob(p) for p in args["<path>"]], []))
    results = lcnn.utils.parmap(work, dirs)

    for d, msAP in zip(dirs, results):
        print(f"{d}: {msAP[0]:2.2f} {msAP[1]:2.2f} {msAP[2]:2.2f}")
