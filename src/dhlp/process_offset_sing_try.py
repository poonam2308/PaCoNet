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

# masks path for color  without unet 2
# MASK_ROOT = "data/pcw_ntest/masks"

# masks path for the cluster  + unet  3
# MASK_ROOT = "data/pcw_test_cls/masks"

# masks path for cluster without unet 4
# MASK_ROOT = "data/pcw_ntest_cls/masks"



# ---- soft toggles ----
USE_MASK = True   # set to False to ignore masks
USE_NMS  = True   # set to False to skip line_nms
# ----------------------

HEATMAP_H, HEATMAP_W = 128, 128          # jmap / lmap size

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_mask_for_index(dataset, index, mask_key="mask"):
    """
    dataset: WireframeDataset
    index:   global index in dataset (0..len(dataset)-1)
    mask_key: key inside the mask npz (e.g. 'mask')

    Returns: [H, W] boolean mask tensor in heatmap coords (e.g. 128x128)
    """
    # label npz path from dataset
    label_path = dataset.filelist[index]          # e.g. data/pcw_test/test/..._label.npz
    base_name = os.path.basename(label_path)      # e.g. image_1_crop_1_BKxJsl_0_label.npz

    # mask npz in separate directory but same file name
    mask_path = os.path.join(MASK_ROOT, base_name)

    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Mask npz not found for index {index}: {mask_path}")

    npz = np.load(mask_path)
    if mask_key not in npz.files:
        raise KeyError(f"Key '{mask_key}' not found in {mask_path}. Available: {npz.files}")

    mask = npz[mask_key]          # shape (128,128), dtype uint8
    npz.close()

    # ensure [H, W]
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    mask_bool = mask > 0          # non-zero is inside
    return torch.from_numpy(mask_bool)  # [H, W], bool


def build_gt_lines_from_meta(meta_i: dict) -> torch.Tensor:
    """
    Build a [N_gt, 2, 2] tensor of GT lines from a single
    sample's meta dict produced by WireframeDataset.

    meta_i["junc"]: [N_junc, 2]
    meta_i["Lpos"]: [N_junc+1, N_junc+1] binary adjacency
    """
    junc = meta_i["junc"]        # (N_junc, 2)
    Lpos = meta_i["Lpos"]        # (N_junc+1, N_junc+1)
    device = junc.device

    N_junc = junc.shape[0]
    lines = []
    for u in range(N_junc):
        for v in range(u + 1, N_junc):
            if Lpos[u, v] > 0:
                lines.append(torch.stack([junc[u], junc[v]], dim=0))

    if len(lines) == 0:
        print("the number of lines ", len(lines))
        return torch.empty(0, 2, 2, device=device)

    return torch.stack(lines, dim=0)
def sample_points_on_line_heatmap(line, n_points):
    """
    line: [2, 2] (y, x) in heatmap coords
    returns: [n_points, 2] (y, x) in heatmap coords
    """
    p0 = line[0]  # (y, x)
    p1 = line[1]
    t = torch.linspace(0.0, 1.0, n_points, device=line.device)[:, None]  # [n,1]
    pts = (1.0 - t) * p0[None, :] + t * p1[None, :]                      # [n,2]
    return pts


def filter_lines_with_mask_heatmap(lines, scores, mask,
                                   min_frac_inside=0.5, n_samples=16):
    """
    lines:  [N, 2, 2] in heatmap coords (y,x)
    scores: [N]
    mask:   [H, W] bool in SAME heatmap coords (e.g. 128x128)

    Returns: filtered_lines, filtered_scores, keep_idx
    """
    if lines.numel() == 0:
        return lines.new_zeros((0, 2, 2)), scores.new_zeros((0,)), torch.empty(0, dtype=torch.long)

    mask = mask.to(dtype=torch.bool)
    H, W = mask.shape
    device = lines.device
    mask = mask.to(device)

    keep = []
    for i in range(lines.shape[0]):
        pts = sample_points_on_line_heatmap(lines[i], n_samples)  # [n,2] in heatmap coords
        ys = pts[:, 0].round().long()
        xs = pts[:, 1].round().long()
        ys.clamp_(0, H - 1)
        xs.clamp_(0, W - 1)
        inside = mask[ys, xs]
        frac_inside = inside.float().mean().item()
        if frac_inside >= min_frac_inside:
            keep.append(i)

    if len(keep) == 0:
        return lines.new_zeros((0, 2, 2)), scores.new_zeros((0,)), torch.empty(0, dtype=torch.long)

    keep_idx = torch.tensor(keep, dtype=torch.long, device=device)
    return lines[keep_idx], scores[keep_idx], keep_idx


def segment_distance(l1, l2):
    """
    Average endpoint distance between two segments, best endpoint ordering.
    l1, l2: [2,2] (y,x)
    returns: scalar float
    """
    p1, p2 = l1[0], l1[1]
    q1, q2 = l2[0], l2[1]
    d1 = (p1 - q1).norm() + (p2 - q2).norm()
    d2 = (p1 - q2).norm() + (p2 - q1).norm()
    return float(min(d1, d2) / 2.0)


def line_nms(lines, scores, dist_thresh=2.0):
    """
    Greedy line-level NMS in heatmap coords.

    lines:  [N, 2, 2]
    scores: [N]
    returns: kept_lines, kept_scores, keep_idx
    """
    if lines.numel() == 0:
        return lines, scores, torch.empty(0, dtype=torch.long)

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
    keep_idx = order[keep_rel]  # back to original indices
    return kept_lines, kept_scores, keep_idx


def match_lines(pred: torch.Tensor,
                gt: torch.Tensor,
                tau: float) -> Tuple[int, int, int]:
    """
    One-to-one matching between predicted and GT line segments.

    pred: [N_pred, 2, 2]  (y, x) in the SAME coordinate system as gt
    gt:   [N_gt,   2, 2]
    tau:  max allowed average endpoint error (in pixels / grid units)

    Returns:
        TP, FP, FN
    """
    Np = pred.shape[0]
    Ng = gt.shape[0]

    if Np == 0 and Ng == 0:
        return 0, 0, 0
    if Np == 0:
        return 0, 0, Ng
    if Ng == 0:
        return 0, Np, 0

    # pred endpoints
    P1 = pred[:, 0, :]  # [Np, 2]
    P2 = pred[:, 1, :]  # [Np, 2]

    # gt endpoints
    G1 = gt[:, 0, :]    # [Ng, 2]
    G2 = gt[:, 1, :]    # [Ng, 2]

    # expand for broadcasting
    P1e = P1[:, None, :]   # [Np, 1, 2]
    P2e = P2[:, None, :]   # [Np, 1, 2]
    G1e = G1[None, :, :]   # [1, Ng, 2]
    G2e = G2[None, :, :]   # [1, Ng, 2]

    d1 = (P1e - G1e).norm(dim=-1) + (P2e - G2e).norm(dim=-1)
    d2 = (P1e - G2e).norm(dim=-1) + (P2e - G1e).norm(dim=-1)
    d = torch.min(d1, d2) / 2.0  # average endpoint error

    # candidate pairs within threshold
    cand = torch.nonzero(d <= tau, as_tuple=False)
    if cand.numel() == 0:
        TP = 0
        FP = Np
        FN = Ng
        return TP, FP, FN

    errors = d[cand[:, 0], cand[:, 1]]
    order = torch.argsort(errors)

    matched_pred = torch.zeros(Np, dtype=torch.bool, device=pred.device)
    matched_gt = torch.zeros(Ng, dtype=torch.bool, device=pred.device)
    TP = 0

    for k in order:
        i = cand[k, 0].item()
        j = cand[k, 1].item()
        if (not matched_pred[i]) and (not matched_gt[j]):
            matched_pred[i] = True
            matched_gt[j] = True
            TP += 1

    FP = (~matched_pred).sum().item()
    FN = (~matched_gt).sum().item()
    return TP, FP, FN


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
        WireframeDataset(rootdir=C.io.datadir, split="test"),
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
                    mask_i = get_mask_for_index(loader.dataset, global_idx)  # [128,128] bool

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
