#!/usr/bin/env python3
"""
Compute line detection metrics (TP / FP / FN) for L-CNN.

IMPORTANT CHANGE:
- The dataset contains multiple "cat images" per same crop.
- We run the model on each cat image, then MERGE predictions for all cat images
  that belong to the same crop (e.g., image_1_crop_1_*), and compute SAP per crop.

Usage:
    process_lines_metric_merge_per_crop.py [options] <yaml-config> <checkpoint>
    process_lines_metric_merge_per_crop.py (-h | --help)

Arguments:
    <yaml-config>   Path to the yaml hyper-parameter file.
    <checkpoint>    Path to the trained checkpoint (.pth.tar).

Options:
    -h --help                     Show this screen.
    -d --devices <devices>        Comma separated GPU devices [default: 0]
"""


### this is used when i group the esult fo different cate and use peak and dbscan mdhlp on single cat and group them.
import os
import sys
import random
import pprint
from collections import defaultdict
from typing import Tuple, Dict, List

import numpy as np
import torch
from docopt import docopt
import scipy.io as sio


# Make sure the project root (where "src/dhlp/lcnn" lives) is on PYTHONPATH.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)


import src.dhlp.lcnn
from src.dhlp.lcnn.utils import recursive_to
from src.dhlp.lcnn.config import C, M
from src.dhlp.lcnn.datasets import WireframeDataset, collate
from src.dhlp.lcnn.models.line_vectorizer import LineVectorizer
from src.dhlp.lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
from src.dhlp.lcnn.models.HT import hough_transform
from src.dhlp.process_utils import segment_distance, filter_lines_with_mask_heatmap, build_gt_lines_from_meta, match_lines, \
    set_seed, crop_key_from_label_path

from src.dhlp.sap_metric import LineSegmentSAPMetric

# ------------------------------------------------------------
# YOU: set mask root (same as your current file)
# ------------------------------------------------------------
# MASK_ROOT = "data/pcw_ntest_cls/masks"

# # masks path for the color + unet 1
# MASK_ROOT = "data/pcw_test/masks"

# masks path for the cluster  + unet  2
MASK_ROOT = "data/pcw_test_cls/masks"

# ---- soft toggles (same style as yours) ----
USE_MASK = True
USE_NMS  = True
# -------------------------------------------

HEATMAP_H, HEATMAP_W = 128, 128

def line_nms(lines, scores, dist_thresh=2.0):
    if lines.numel() == 0:
        return lines, scores

    scores, order = torch.sort(scores, descending=True)
    lines = lines[order]

    keep_rel = []
    for i in range(lines.shape[0]):
        li = lines[i]
        suppress = False
        for k in keep_rel:
            lk = lines[k]
            d = segment_distance(li, lk)
            if d < dist_thresh:
                suppress = True
                break
        if not suppress:
            keep_rel.append(i)

    keep_rel = torch.tensor(keep_rel, dtype=torch.long, device=lines.device)
    kept_lines = lines[keep_rel]
    kept_scores = scores[keep_rel]
    return kept_lines, kept_scores



def get_mask_for_label_path(label_path: str, mask_key="mask") -> torch.Tensor:
    base = os.path.basename(label_path)  # e.g. image_1000_crop_1_0VSK_0_label.npz

    candidates = [
        os.path.join(MASK_ROOT, base),  # same name
        os.path.join(MASK_ROOT, base.replace("_label.npz", ".npz")),  # drop _label
        os.path.join(MASK_ROOT, base.replace("_label.npz", "_mask.npz")),  # _mask suffix
        os.path.join(MASK_ROOT, base.replace("_label.npz", "")),  # raw
    ]

    mask_path = None
    for p in candidates:
        if os.path.isfile(p):
            mask_path = p
            break

    if mask_path is None:
        raise FileNotFoundError(
            "Mask npz not found. Tried:\n" + "\n".join(candidates)
        )

    npz = np.load(mask_path)
    if mask_key not in npz.files:
        raise KeyError(f"Key '{mask_key}' not found in {mask_path}. Available: {npz.files}")

    mask = npz[mask_key]
    npz.close()

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    return torch.from_numpy(mask > 0)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    set_seed(0)
    args = docopt(__doc__)
    config_file = args["<yaml-config>"]

    # 1. Load config
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    pprint.pprint(C, indent=4)

    # 2. Device
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 3. Load / build vote_index for Hough transform
    if os.path.isfile(C.io.vote_index):
        vote_index = sio.loadmat(C.io.vote_index)["vote_index"]
    else:
        vote_index = hough_transform(rows=128, cols=128, theta_res=3, rho_res=1)
        sio.savemat(C.io.vote_index, {"vote_index": vote_index})
    vote_index = torch.from_numpy(vote_index).float().contiguous().to(device)
    print("vote_index:", vote_index.shape)

    # 4. Build model
    if M.backbone == "stacked_hourglass":
        model = src.dhlp.lcnn.models.hg(
            depth=M.depth,
            head=MultitaskHead,
            num_stacks=M.num_stacks,
            num_blocks=M.num_blocks,
            num_classes=sum(sum(M.head_size, [])),
            vote_index=vote_index,
        )
    else:
        raise NotImplementedError(f"Unknown backbone {M.backbone}")

    model = MultitaskLearner(model)
    model = LineVectorizer(model)

    # 5. Load checkpoint
    checkpoint = torch.load(args["<checkpoint>"], map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print("Loaded checkpoint from", args["<checkpoint>"])

    # 6. Data loader
    loader = torch.utils.data.DataLoader(
        WireframeDataset(rootdir=C.io.datadir, split="test"),
        shuffle=False,
        batch_size=M.batch_size,
        collate_fn=collate,
        num_workers=C.io.num_workers if os.name != "nt" else 0,
        pin_memory=True,
    )

    # 7. Metric
    metric = LineSegmentSAPMetric(thresholds=(5.0, 10.0, 15.0))

    # tau threshold
    tau = float(getattr(M, "line_match_thresh", 5.0))
    print(f"Using line-matching threshold tau = {tau} (heatmap coordinates)")

    # ------------------------------------------------------------
    # NEW: accumulate per-crop predictions, not per-image
    # ------------------------------------------------------------
    crop_pred_lines = defaultdict(list)
    crop_pred_scores = defaultdict(list)
    crop_gt_lines = defaultdict(list)  # <-- IMPORTANT

    global_idx = 0

    with torch.no_grad():
        for batch_idx, (image, meta, target) in enumerate(loader):
            input_dict = {
                "image": recursive_to(image, device),
                "meta": recursive_to(meta, device),
                "target": recursive_to(target, device),
                "mode": "validation",
            }
            out = model(input_dict)
            H = out["preds"]
            B = image.shape[0]

            for i in range(B):
                label_path = loader.dataset.filelist[global_idx]
                crop_key = crop_key_from_label_path(label_path)

                # raw predictions (heatmap coords)
                lines_i = H["lines"][i].detach().cpu()
                scores_i = H["score"][i].detach().cpu()

                # optional mask filter (per cat-image)
                if USE_MASK:
                    mask_i = get_mask_for_label_path(label_path)
                    lines_i, scores_i,_ = filter_lines_with_mask_heatmap(
                        lines_i, scores_i, mask_i,
                        min_frac_inside=0.5,
                        n_samples=16,
                    )

                # DO NOT NMS here (best practice) — do it AFTER merging per crop
                crop_pred_lines[crop_key].append(lines_i)
                crop_pred_scores[crop_key].append(scores_i)

                # store GT once per crop
                gt_lines_i = build_gt_lines_from_meta(meta[i]).detach().cpu()
                crop_gt_lines[crop_key].append(gt_lines_i)

                global_idx += 1

    # ------------------------------------------------------------
    # Evaluate per crop (merged cats)
    # ------------------------------------------------------------
    total_TP = total_FP = total_FN = 0
    n_crops = 0

    for crop_key in sorted(crop_gt_lines.keys()):
        # merge GT across cats
        gt_lines = torch.cat(crop_gt_lines[crop_key], dim=0) if len(crop_gt_lines[crop_key]) else torch.empty(0, 2, 2)

        # dedupe merged GT (important)
        gt_scores_dummy = torch.ones(gt_lines.shape[0])
        gt_lines, _ = line_nms(gt_lines, gt_scores_dummy, dist_thresh=1.0)

        if len(crop_pred_lines[crop_key]) > 0:
            merged_lines = torch.cat(crop_pred_lines[crop_key], dim=0)
            merged_scores = torch.cat(crop_pred_scores[crop_key], dim=0)
        else:
            merged_lines = torch.empty(0, 2, 2)
            merged_scores = torch.empty(0)

        # NMS AFTER merging all cats of this crop
        if USE_NMS:
            merged_lines, merged_scores = line_nms(merged_lines, merged_scores, dist_thresh=2.0)

        TP, FP, FN = match_lines(merged_lines, gt_lines, tau=tau)
        total_TP += TP
        total_FP += FP
        total_FN += FN
        n_crops += 1

        print(
            f"CROP {n_crops:05d} {crop_key}: TP={TP:3d}, FP={FP:3d}, FN={FN:3d} "
            f"(running totals: TP={total_TP}, FP={total_FP}, FN={total_FN})"
        )

        # SAP expects "one image" => we treat each CROP as one unit
        metric.add_image(merged_lines, merged_scores, gt_lines)

    # final metrics
    precision = total_TP / (total_TP + total_FP + 1e-8)
    recall = total_TP / (total_TP + total_FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("\n===== MERGED-CROP Line Detection Metrics =====")
    print(f"Total crops:     {n_crops}")
    print(f"Total GT lines:  {total_TP + total_FN}")
    print(f"Total pred lines:{total_TP + total_FP}")
    print(f"TP: {total_TP}  FP: {total_FP}  FN: {total_FN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\n===== MERGED-CROP SAP =====")
    sap = metric.compute_sap()
    print("sAP5, sAP10, sAP15:", sap[5.0], sap[10.0], sap[15.0])
    print("sAP5%, sAP10%, sAP15%:",
          100.0 * sap[5.0], 100.0 * sap[10.0], 100.0 * sap[15.0])


if __name__ == "__main__":
    main()
