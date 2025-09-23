import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# -------------------------------------------------------------------
# 🔹 Shared helper functions
# -------------------------------------------------------------------

def ensure_left_to_right(lines):
    """Ensure each line is oriented left-to-right by x-coordinate."""
    corrected_lines = []
    for line in lines:
        (x0, y0), (x1, y1) = line
        if x0 > x1:  # Swap if not left to right
            line = [[x1, y1], [x0, y0]]
        corrected_lines.append(line)
    return np.array(corrected_lines)


def filter_lines_with_mask(lines, mask):
    valid_lines = []
    h, w = mask.shape
    for line in lines:
        (a, b) = line.astype(int)
        # y0, x0 = a[0], a[1]
        # y1, x1 = b[0], b[1]

        x0, y0 = a[0], a[1]
        x1, y1 = b[0], b[1]

        if 0 <= y0 < h and 0 <= x0 < w and 0 <= y1 < h and 0 <= x1 < w:
            if mask[y0, x0] == 1 and mask[y1, x1] == 1:
                valid_lines.append(line)
    return np.array(valid_lines)



def remove_duplicate_lines(lines, eps=1.0):
    """Remove near-identical lines within eps distance."""
    unique_lines = []
    for line in lines:
        if all(np.linalg.norm(line - ul) > eps for ul in unique_lines):
            unique_lines.append(line)
    return np.array(unique_lines)

import numpy as np

def _angle_deg(line):
    (x0,y0),(x1,y1) = line
    ang = np.degrees(np.arctan2((y1-y0), (x1-x0)))  # [-180,180]
    # Treat directions that differ by 180° as the same (undirected lines)
    ang = (ang + 180.0) % 180.0
    return ang

def _pt_seg_dist(p, a, b):
    # distance from point p to segment ab
    ap = p - a
    ab = b - a
    denom = np.dot(ab, ab)
    if denom <= 1e-12:
        return np.linalg.norm(ap)
    t = np.clip(np.dot(ap, ab) / denom, 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj)

def _sym_endpoint_to_segment_dist(lineA, lineB):
    a0,a1 = lineA
    b0,b1 = lineB
    d1 = max(_pt_seg_dist(a0, b0, b1), _pt_seg_dist(a1, b0, b1))
    d2 = max(_pt_seg_dist(b0, a0, a1), _pt_seg_dist(b1, a0, a1))
    return max(d1, d2)

def _overlap_along_dir(lineA, lineB):
    # project endpoints onto a common unit direction (avg direction of A and B)
    a0,a1 = lineA; b0,b1 = lineB
    va = a1 - a0
    vb = b1 - b0
    # choose the longer as ref if one is near-zero
    v  = va if np.dot(va,va) >= np.dot(vb,vb) else vb
    nv = np.linalg.norm(v)
    if nv < 1e-9:
        return 0.0
    u = v / nv
    # pick a common origin to keep projections comparable
    o = (a0 + a1 + b0 + b1) / 4.0
    def proj_interval(p, q):
        t0 = np.dot(p - o, u)
        t1 = np.dot(q - o, u)
        return (min(t0,t1), max(t0,t1))
    A = proj_interval(a0,a1)
    B = proj_interval(b0,b1)
    return max(0.0, min(A[1], B[1]) - max(A[0], B[0]))

def msTPFP_oriented(line_pred, line_gt, dist_thresh=10.0, ang_thresh_deg=10.0, min_overlap=5.0, debug=False):
    """
    Geometry-aware matching:
      - Reject pairs if angle diff > ang_thresh_deg
      - Reject pairs if symmetric endpoint→segment distance > dist_thresh
      - Reject pairs if 1-D overlap along direction < min_overlap
    Greedy one-to-one matching on remaining pairs (lowest distance first).
    Returns tp, fp (float arrays of shape [N_pred])
    """
    nP, nG = len(line_pred), len(line_gt)
    if nP == 0 or nG == 0:
        return np.zeros(nP, float), np.ones(nP, float)  # no GT → all FPs

    # Precompute angles
    angP = np.array([_angle_deg(l) for l in line_pred])
    angG = np.array([_angle_deg(l) for l in line_gt])

    # Build a cost matrix using symmetric endpoint distance; inf where gated out
    cost = np.full((nP, nG), np.inf, dtype=np.float32)
    for i in range(nP):
        for j in range(nG):
            # angle gate
            adiff = abs(angP[i] - angG[j])
            adiff = min(adiff, 180.0 - adiff)  # shortest undirected difference
            if adiff > ang_thresh_deg:
                continue
            # distance (symmetric endpoint→segment)
            d = _sym_endpoint_to_segment_dist(line_pred[i], line_gt[j])
            if d > dist_thresh:
                continue
            # overlap gate
            ov = _overlap_along_dir(line_pred[i], line_gt[j])
            if ov < min_overlap:
                continue
            cost[i, j] = d  # lower is better

    # Greedy assignment: repeatedly pick smallest cost and mark both as used
    tp = np.zeros(nP, float)
    fp = np.ones(nP, float)  # start as FP; flip to TP when matched
    used_gt = np.zeros(nG, dtype=bool)

    # Flatten and sort valid pairs by cost
    pairs = np.argwhere(np.isfinite(cost))
    if pairs.size > 0:
        order = np.argsort(cost[pairs[:,0], pairs[:,1]])
        for idx in order:
            i, j = pairs[idx]
            if tp[i] == 1.0 or used_gt[j]:
                continue
            # assign this pair
            tp[i] = 1.0
            fp[i] = 0.0
            used_gt[j] = True
            if debug:
                print(f"[TP] Pred {i} ↔ GT {j} | dist={cost[i,j]:.2f}")

    # Unmatched predictions remain FP
    if debug:
        for i in range(nP):
            if fp[i] == 1.0:
                # show nearest (even if gated) for context
                j_best = np.argmin(cost[i]) if np.any(np.isfinite(cost[i])) else None
                reason = "no gated match" if j_best is None else f"lost to GT {j_best}"
                print(f"[FP] Pred {i}: {reason}")
    return tp, fp


def compute_ap(tp, fp, num_gt):
    """
    Average Precision with recall normalized by the number of ground-truth lines.

    Args:
        tp (array-like): 1 for each prediction counted as a true positive (ranked order).
        fp (array-like): 1 for each prediction counted as a false positive (ranked order).
        num_gt (int):     Count of ground-truth lines you are evaluating against
                          (after any GT filtering you applied earlier).
    Returns:
        float: AP in [0, 1].
    """
    tp = np.asarray(tp, dtype=np.float32)
    fp = np.asarray(fp, dtype=np.float32)

    if num_gt <= 0:
        # No ground truth → nothing to recall; define AP as 0.0
        return 0.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / float(num_gt)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1.0)

    # precision envelope (monotonic non-increasing)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # integrate PR curve where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap

def msTPFP(line_pred, line_gt, threshold, debug=False):
    """
    Compute true positives (TP) and false positives (FP) for line matching.
    Matching is based on midpoint distance between lines.

    Args:
        line_pred: (N_pred, 2, 2) array of predicted lines
        line_gt:   (N_gt, 2, 2) array of ground truth lines
        threshold: distance threshold (in pixels)
        debug:     if True, prints TP/FP decisions

    Returns:
        tp: (N_pred,) array of 0/1 for true positives
        fp: (N_pred,) array of 0/1 for false positives
    """
    # Compute midpoints of all lines
    pred_mid = line_pred.mean(axis=1)  # (N_pred, 2)
    gt_mid = line_gt.mean(axis=1)  # (N_gt, 2)

    # Pairwise midpoint distances
    diff = np.linalg.norm(pred_mid[:, None, :] - gt_mid[None, :, :], axis=-1)  # (N_pred, N_gt)

    # Best GT match for each prediction
    choice = np.argmin(diff, 1)
    dist = np.min(diff, 1)

    # Bookkeeping
    hit = np.zeros(len(line_gt), bool)
    tp = np.zeros(len(line_pred), float)
    fp = np.zeros(len(line_pred), float)

    for i in range(len(line_pred)):
        gt_idx = choice[i]
        if dist[i] < threshold and not hit[gt_idx]:
            hit[gt_idx] = True
            tp[i] = 1
            if debug:
                print(f"[TP] Pred {i} → GT {gt_idx}, dist={dist[i]:.2f}, threshold={threshold}")
                print(f"     Pred line: {line_pred[i]}")
                print(f"     GT line:   {line_gt[gt_idx]}")
        else:
            fp[i] = 1
            if debug:
                reason = "duplicate" if dist[i] < threshold else "too far"
                print(f"[FP] Pred {i} → GT {gt_idx}, dist={dist[i]:.2f}, reason={reason}")
                print(f"     Pred line: {line_pred[i]}")
                print(f"     GT line:   {line_gt[gt_idx]}")

    return tp, fp

import numpy as np

def _sample_segment(line, step_px=3.0, min_pts=5, max_pts=200):
    """
    Uniformly sample points along a line segment.
    line: ((x0,y0),(x1,y1))
    """
    (x0, y0), (x1, y1) = line
    v = np.array([x1 - x0, y1 - y0], dtype=float)
    L = np.hypot(v[0], v[1])
    if L < 1e-9:
        return np.array([[x0, y0]], dtype=float)
    n = int(np.ceil(L / max(step_px, 1e-6)))
    n = int(np.clip(n, min_pts, max_pts))
    t = np.linspace(0.0, 1.0, n)
    return np.stack([x0 + t * v[0], y0 + t * v[1]], axis=1)  # (n,2)

def msTPFP_sampels(line_pred, line_gt, threshold, debug=False):
    """
    Compute TP/FP for line matching using sampled point clouds.
    For each pred–GT pair, we sample points on both segments and take the
    minimum distance between any pair of sampled points as the match score.

    Args:
        line_pred: (N_pred, 2, 2) array of predicted lines
        line_gt:   (N_gt,  2, 2) array of ground truth lines
        threshold: distance threshold (in pixels) applied to the min sample distance
        debug:     if True, prints TP/FP decisions

    Returns:
        tp: (N_pred,) array of {0,1} for true positives
        fp: (N_pred,) array of {0,1} for false positives
    """
    Np, Ng = len(line_pred), len(line_gt)
    tp = np.zeros(Np, float)
    fp = np.zeros(Np, float)

    if Np == 0:
        return tp, fp
    if Ng == 0:
        fp[:] = 1.0
        if debug:
            print("[WARN] No GT lines; all predictions are FP.")
        return tp, fp

    # --- Tunables (kept inside to preserve the original function signature) ---
    STEP_PX = 3.0   # sampling stride in pixels
    MIN_PTS = 5     # ensure at least a few samples even for short segments
    MAX_PTS = 200   # cap to avoid O(n^2) blowups on very long lines

    # Pre-sample points for all lines
    pred_samples = [ _sample_segment(l, STEP_PX, MIN_PTS, MAX_PTS) for l in line_pred ]
    gt_samples   = [ _sample_segment(l, STEP_PX, MIN_PTS, MAX_PTS) for l in line_gt   ]

    # Build pairwise min sample-to-sample distance matrix (Np x Ng)
    # dist_ij = min_{a in P_i, b in G_j} ||a - b||
    dist = np.full((Np, Ng), np.inf, dtype=float)
    for i in range(Np):
        A = pred_samples[i]  # (Na,2)
        # Precompute norms for efficiency if needed (we’ll do direct vectorized diffs)
        for j in range(Ng):
            B = gt_samples[j]  # (Nb,2)
            # (Na,1,2) - (1,Nb,2) -> (Na,Nb,2) -> (Na,Nb)
            d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
            dist[i, j] = float(np.sqrt(d2.min())) if d2.size else np.inf

    # Best GT match for each prediction (same logic as your original)
    choice = np.argmin(dist, axis=1)
    mind   = np.min(dist, axis=1)

    hit = np.zeros(Ng, dtype=bool)
    for i in range(Np):
        gt_idx = choice[i]
        if mind[i] < threshold and not hit[gt_idx]:
            hit[gt_idx] = True
            tp[i] = 1.0
            if debug:
                print(f"[TP] Pred {i} → GT {gt_idx}, min_sample_dist={mind[i]:.2f}, threshold={threshold}")
                print(f"     Pred line: {line_pred[i]}")
                print(f"     GT line:   {line_gt[gt_idx]}")
        else:
            fp[i] = 1.0
            if debug:
                reason = "duplicate" if mind[i] < threshold else "too far"
                print(f"[FP] Pred {i} → GT {gt_idx}, min_sample_dist={mind[i]:.2f}, reason={reason}")
                print(f"     Pred line: {line_pred[i]}")
                print(f"     GT line:   {line_gt[gt_idx]}")

    return tp, fp




# -------------------------------------------------------------------
#   New array-based version (no .npz needed)
# -------------------------------------------------------------------


def process_line_detection_arrays(ground_truth_lines, predicted_lines, mask, threshold=10, eps=1.0):
    print("GT lines before:", ground_truth_lines.shape)
    print("Pred lines before:", predicted_lines.shape)
    predicted_lines = ensure_left_to_right(predicted_lines)
    ground_truth_lines = ensure_left_to_right(ground_truth_lines)
    filtered_lines = filter_lines_with_mask(predicted_lines, mask)
    print("Filtered pred lines after mask :", filtered_lines.shape)
    filtered_lines = remove_duplicate_lines(filtered_lines, eps)
    filtered_lines = np.round(filtered_lines, 2)
    print("Filtered pred lines:", filtered_lines.shape)

    # 🔹 DEBUG: show a few examples
    # print("\n--- Sample GT lines (first 3) ---")
    # print(ground_truth_lines[:1])
    # print("\n--- Sample Pred lines (first 3 before filtering) ---")
    # print(predicted_lines[:1])
    # print("\n--- Sample Pred lines (first 3 after filtering) ---")
    # print(filtered_lines[:1])

    if len(filtered_lines) == 0 or len(ground_truth_lines) == 0:
        print("⚠️ Skipping: no valid lines to compare")
        return 0.0

    tp, fp = msTPFP_sampels(filtered_lines, ground_truth_lines, threshold, debug=False)
    num_gt = len(ground_truth_lines)
    print("TP:", tp.sum(), "FP:", fp.sum())
    return compute_ap(tp, fp,num_gt) * 100




