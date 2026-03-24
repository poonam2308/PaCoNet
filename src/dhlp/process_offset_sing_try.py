#!/usr/bin/env python3
"""
Compute line detection metrics (TP / FP / FN) for L-CNN.

Usage:
    process_lines_metric.py [options] <yaml-config> <checkpoint>
    process_lines_metric.py (-h | --help)

Arguments:
    <yaml-config>   Path to the yaml hyper-parameter file.
    <checkpoint>    Path to the trained checkpoint (.pth.tar).

Options:
    -h --help                     Show this screen.
    -d --devices <devices>        Comma separated GPU devices [default: 0]
"""

import os
import sys
import random
import pprint

import numpy as np
import torch
from typing import Tuple
from docopt import docopt
import scipy.io as sio

# Make sure the project root (where "src/dhlp/lcnn" lives) is on PYTHONPATH.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.dhlp.process_utils import set_seed, get_mask_for_index, filter_lines_with_mask_heatmap, line_nms, \
    build_gt_lines_from_meta, match_lines

import src.dhlp.lcnn  # noqa: F401
from src.dhlp.lcnn.utils import recursive_to
from src.dhlp.lcnn.config import C, M
from src.dhlp.lcnn.datasets import WireframeDataset, collate
from src.dhlp.lcnn.models.line_vectorizer import LineVectorizer
from src.dhlp.lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
from src.dhlp.lcnn.models.HT import hough_transform

from src.dhlp.sap_metric import LineSegmentSAPMetric  # wherever you put the class

metric = LineSegmentSAPMetric(thresholds=(5.0, 10.0, 15.0))


# ------------------------------------------------------------
# Line matching utilities
# ------------------------------------------------------------

# directory containing your mask npz files (same names as label npz)
# e.g. if labels are in:  data/pcw_test/test/*.npz
# and masks are in:       data/pcw_test_masks/test/*.npz

# # masks path for the color + unet 1
MASK_ROOT = "data/pcw_test/masks"

# masks path for the cluster  + unet  2
# MASK_ROOT = "data/pcw_test_cls/masks"

# # masks path for color  without unet 3
# MASK_ROOT = "data/pcw_ntest/masks"

# masks path for cluster without unet 4
# MASK_ROOT = "data/pcw_ntest_cls/masks"

# # all cat in crops
# MASK_ROOT = "data/pcw_crops_test/masks"



# ---- soft toggles ----
USE_MASK = True   # set to False to ignore masks
USE_NMS  = True   # set to False to skip line_nms
# ----------------------

HEATMAP_H, HEATMAP_W = 128, 128          # jmap / lmap size


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

    # 2. Set random seeds
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # 3. Device
    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Using", torch.cuda.device_count(), "GPU(s)")
    else:
        print("CUDA not available, using CPU")
    device = torch.device(device_name)

    # 4. Load / build vote_index for Hough transform
    if os.path.isfile(C.io.vote_index):
        vote_index = sio.loadmat(C.io.vote_index)["vote_index"]
    else:
        vote_index = hough_transform(rows=128, cols=128, theta_res=3, rho_res=1)
        sio.savemat(C.io.vote_index, {"vote_index": vote_index})
    vote_index = torch.from_numpy(vote_index).float().contiguous().to(device)
    print("vote_index:", vote_index.shape)

    # 5. Build model (same as in train.py)
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

    # 6. Load checkpoint
    checkpoint = torch.load(args["<checkpoint>"], map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print("Loaded checkpoint from", args["<checkpoint>"])

    # 7. Data loader (change split to 'valid' / 'test' as you like)
    loader = torch.utils.data.DataLoader(
        WireframeDataset(rootdir=C.io.datadir, split="low_level"),
        shuffle=False,
        batch_size=M.batch_size,
        collate_fn=collate,
        num_workers=C.io.num_workers if os.name != "nt" else 0,
        pin_memory=True,
    )

    # 8. Evaluation loop
    # Threshold in heatmap pixels; you can override via config (M.line_match_thresh)
    tau = float(getattr(M, "line_match_thresh", 5.0))
    print(f"Using line-matching threshold tau = {tau} (heatmap coordinates)")

    total_TP = 0
    total_FP = 0
    total_FN = 0
    n_images = 0

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

            # global index over the whole dataset
            # (don't depend on batch size; just count)
            for i in range(B):
                # compute global index in dataset
                # we can use a separate counter outside the loop, but easiest:
                global_idx = batch_idx * loader.batch_size + i

                # 1) raw predictions in heatmap coords
                lines_i = H["lines"][i].detach().cpu()  # [n_out_line, 2, 2]
                scores_i = H["score"][i].detach().cpu()  # [n_out_line]

                # 2–3) optional mask filtering
                if USE_MASK:
                    mask_i = get_mask_for_index(loader.dataset, global_idx, MASK_ROOT)  # [128,128] bool

                    lines_i, scores_i, _ = filter_lines_with_mask_heatmap(
                        lines_i, scores_i, mask_i,
                        min_frac_inside=0.5,
                        n_samples=16,
                    )
                # else: keep raw lines_i / scores_i

                # 4) optional: line-level NMS to reduce overlapping lines
                if USE_NMS:
                    lines_i, scores_i, _ = line_nms(
                        lines_i, scores_i,
                        dist_thresh=2.0,
                    )
                # else: keep whatever we currently have

                # 5) evaluate these filtered lines against GT
                meta_i = meta[i]
                gt_lines = build_gt_lines_from_meta(meta_i).detach().cpu()

                pred_lines = lines_i  # already CPU
                TP, FP, FN = match_lines(pred_lines, gt_lines, tau=tau)
                total_TP += TP
                total_FP += FP
                total_FN += FN
                n_images += 1

                print(
                    f"Image {n_images:05d}: TP={TP:3d}, FP={FP:3d}, FN={FN:3d} "
                    f"(running totals: TP={total_TP}, FP={total_FP}, FN={total_FN})"
                )

                metric.add_image(lines_i, scores_i, gt_lines)

    # 9. Final metrics
    precision = total_TP / (total_TP + total_FP + 1e-8)
    recall = total_TP / (total_TP + total_FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("\n===== Line Detection Metrics =====")
    print(f"Total images:   {n_images}")
    print(f"Total GT lines: {total_TP + total_FN}")
    print(f"Total pred lines: {total_TP + total_FP}")
    print(f"TP: {total_TP}  FP: {total_FP}  FN: {total_FN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print( "\n =============== sap===============")

    # After the loop:
    sap = metric.compute_sap()
    print("sAP5, sAP10, sAP15:", sap[5.0], sap[10.0], sap[15.0])
    print("sAP5%, sAP10%, sAP15%:",
          100.0 * sap[5.0], 100.0 * sap[10.0], 100.0 * sap[15.0])

if __name__ == "__main__":
    main()
