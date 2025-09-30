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

def msTPFP_endpt_mae_sequential(line_pred, line_gt, mae_thresh_px=5.0, debug=False):
    """
    Sequential greedy TP/FP using endpoint MAE.
    For each predicted line in order, choose the GT with the minimum MAE (order-invariant endpoints).
    If that min MAE <= threshold and the GT is unused, it's a TP; otherwise FP.

    Args:
        line_pred: (N_pred, 2, 2) float array  [[(x0,y0),(x1,y1)], ...]
        line_gt:   (N_gt,   2, 2) float array
        mae_thresh_px: float threshold on the MAE over (x0,y0,x1,y1)
        debug: print allocation details
    Returns:
        tp: (N_pred,) float in {0,1}
        fp: (N_pred,) float in {0,1}
    """
    Np = len(line_pred)
    Ng = len(line_gt)
    tp = np.zeros(Np, dtype=float)
    fp = np.ones(Np, dtype=float)
    if Np == 0:
        return tp, fp if Ng > 0 else (tp, fp)  # both zeros if no preds

    if Ng == 0:
        return tp, np.ones(Np, dtype=float)

    used_gt = np.zeros(Ng, dtype=bool)

    # small helper: order-invariant endpoint MAE
    def pair_mae(p, g):
        # p, g: shape (2,2)
        p12 = p.reshape(-1)
        g12 = g.reshape(-1)
        g21 = g[::-1].reshape(-1)
        mae_straight = np.mean(np.abs(p12 - g12))
        mae_swapped = np.mean(np.abs(p12 - g21))
        return min(mae_straight, mae_swapped)

    for i in range(Np):
        p = line_pred[i].astype(float)
        # compute MAE to every *unused* GT
        best_j = None
        best_c = np.inf
        for j in range(Ng):
            if used_gt[j]:
                continue
            c = pair_mae(p, line_gt[j].astype(float))
            if c < best_c:
                best_c, best_j = c, j

        if best_j is not None and best_c <= mae_thresh_px:
            tp[i] = 1.0
            fp[i] = 0.0
            used_gt[best_j] = True
            if debug:
                print(f"[TP] Pred {i} -> GT {best_j} | MAE={best_c:.2f}")
        else:
            # either no unused GT left or min cost above threshold
            if debug:
                reason = "no GT left" if best_j is None else f"min MAE {best_c:.2f} > {mae_thresh_px}"
                print(f"[FP] Pred {i} | {reason}")

    return tp, fp

import numpy as np

def msTPFP_endpt_mae_global(line_pred, line_gt, mae_thresh_px=5.0, debug=False, use_hungarian=False):
    """
    Order-invariant TP/FP using endpoint MAE with one-to-one matching.
    Builds a full (Np x Ng) MAE cost matrix (endpoint order-invariant),
    applies a threshold gate, then matches:
      - if use_hungarian=True and SciPy is available: Hungarian assignment
      - else: greedy over all valid pairs sorted by increasing cost (order-invariant)

    Returns:
        tp: (N_pred,) float in {0,1}
        fp: (N_pred,) float in {0,1}
    """
    Np, Ng = len(line_pred), len(line_gt)
    tp = np.zeros(Np, dtype=float)
    fp = np.ones(Np, dtype=float)

    if Np == 0:
        return tp, (np.ones(0, float) if Ng > 0 else fp)
    if Ng == 0:
        return tp, np.ones(Np, dtype=float)

    # --- helper: endpoint-order-invariant MAE (same logic as your sequential version)
    def pair_mae(p, g):
        p12 = p.reshape(-1)
        g12 = g.reshape(-1)
        g21 = g[::-1].reshape(-1)
        mae_straight = np.mean(np.abs(p12 - g12))
        mae_swapped  = np.mean(np.abs(p12 - g21))
        return mae_straight if mae_straight <= mae_swapped else mae_swapped

    # Build cost matrix with gating by threshold
    cost = np.full((Np, Ng), np.inf, dtype=float)
    for i in range(Np):
        pi = line_pred[i].astype(float)
        for j in range(Ng):
            c = pair_mae(pi, line_gt[j].astype(float))
            if c <= mae_thresh_px:
                cost[i, j] = c  # valid match

    # Nothing valid → all FP
    if not np.isfinite(cost).any():
        if debug:
            print("[INFO] No pred–GT pairs under threshold; all predictions are FP.")
        return tp, fp

    used_p = np.zeros(Np, dtype=bool)
    used_g = np.zeros(Ng, dtype=bool)

    if use_hungarian:
        try:
            from scipy.optimize import linear_sum_assignment
            # Replace inf with a large sentinel; we will drop those after assignment
            big = 1e9
            cost_safe = np.where(np.isfinite(cost), cost, big)
            row_ind, col_ind = linear_sum_assignment(cost_safe)
            for i, j in zip(row_ind, col_ind):
                if used_p[i] or used_g[j]:
                    continue
                c = cost[i, j]
                if np.isfinite(c) and c <= mae_thresh_px:
                    tp[i] = 1.0
                    fp[i] = 0.0
                    used_p[i] = True
                    used_g[j] = True
                    if debug:
                        print(f"[TP] Pred {i} ↔ GT {j} | MAE={c:.2f}")
            # Unmatched remain FP
            return tp, fp
        except Exception as e:
            if debug:
                print(f"[WARN] Hungarian path unavailable ({e}); falling back to greedy.")

    # Greedy over all finite-cost pairs (order-invariant w.r.t. prediction order)
    pairs = np.argwhere(np.isfinite(cost))
    order = np.argsort(cost[pairs[:, 0], pairs[:, 1]], kind="mergesort")  # stable tie-break
    for k in order:
        i, j = pairs[k]
        if used_p[i] or used_g[j]:
            continue
        c = cost[i, j]
        tp[i] = 1.0
        fp[i] = 0.0
        used_p[i] = True
        used_g[j] = True
        if debug:
            print(f"[TP] Pred {i} ↔ GT {j} | MAE={c:.2f}")

    # Others remain FP
    if debug:
        for i in range(Np):
            if not used_p[i]:
                # show nearest even if above threshold, for context
                j_best = np.argmin(cost[i]) if np.any(np.isfinite(cost[i])) else None
                reason = "no valid GT under threshold" if j_best is None else f"lost to GT {j_best}"
                print(f"[FP] Pred {i}: {reason}")
    return tp, fp

# -------------------------------------------------------------------
#   New array-based version (no .npz needed)
# -------------------------------------------------------------------

def greedy_match_by_y_endpoints(
    pred_lines: np.ndarray,
    gt_lines: np.ndarray,
    max_y_mae: float | None = None,
    print_first: int = 5
):
    """
    Greedy GT→Pred assignment using y-endpoint absolute differences.

    Cost for a (pred, gt) pair (after left→right orientation):
        cost = |y0_pred - y0_gt| + |y1_pred - y1_gt|
    (i.e., sum of absolute differences on the y-values at the start/end points).
    If `max_y_mae` is set, a match is accepted only if (cost/2) <= max_y_mae.

    Args:
        pred_lines: (Np, 2, 2) [(x0,y0),(x1,y1)] after your filtering.
        gt_lines:   (Ng, 2, 2) [(x0,y0),(x1,y1)] (can be raw; we orient inside).
        max_y_mae:  optional threshold in px (on the mean abs error over y’s).
        print_first: how many example matches to print.

    Returns:
        matches:   list of dicts {gt_idx, pred_idx, y_err0, y_err1, mae_y}
        mapped_gt: int, number of GTs that got a pred assigned (and passed threshold)
        unused_pred_indices: np.ndarray of remaining prediction indices
    """
    if len(gt_lines) == 0:
        return [], 0, np.arange(len(pred_lines))
    if len(pred_lines) == 0:
        return [], 0, np.array([], dtype=int)

    # Ensure consistent orientation so y0=left endpoint, y1=right endpoint
    from copy import deepcopy
    gt_lr   = ensure_left_to_right(gt_lines)       # in your file
    pred_lr = ensure_left_to_right(pred_lines)     # in your file

    Np, Ng = len(pred_lr), len(gt_lr)
    available = set(range(Np))
    matches = []

    # Pre-extract the y-columns for speed
    pred_y = np.stack([pred_lr[:, 0, 1], pred_lr[:, 1, 1]], axis=1)  # (Np, 2)
    gt_y   = np.stack([gt_lr[:, 0, 1],   gt_lr[:, 1, 1]],   axis=1)  # (Ng, 2)

    for j in range(Ng):
        if not available:
            break
        # compute cost to all available preds for this GT
        ay = np.array(sorted(list(available)))  # stable order
        print("predicted points ", pred_y[ay] , "ground truth points ", gt_y[j])
        diff = np.abs(pred_y[ay] - gt_y[j])     # (Na,2) elementwise |Δy|
        cost = diff.sum(axis=1)                 # sum of |Δy| at start/end
        k = int(np.argmin(cost))                # index into ay
        i_best = int(ay[k])
        yerr0, yerr1 = float(diff[k, 0]), float(diff[k, 1])
        mae_y = 0.5 * (yerr0 + yerr1)

        # threshold gate (optional)
        accepted = True
        if accepted:
            matches.append({
                "gt_idx": j,
                "pred_idx": i_best,
                "y_err0": yerr0,
                "y_err1": yerr1,
                "mae_y": mae_y
            })
            available.remove(i_best)
        # if not accepted, we leave the pred available for other GTs

    # ---- Pretty print a few examples
    if print_first > 0 and matches:
        print(f"\n[greedy_match_by_y_endpoints] Showing first {min(print_first, len(matches))} matches:")
        for m in matches[:print_first]:
            j = m["gt_idx"]; i = m["pred_idx"]
            print(f"  GT {j:>3}  ⇐  Pred {i:>3} | y_errs=({m['y_err0']:.2f}, {m['y_err1']:.2f})  mae_y={m['mae_y']:.2f}")

    mapped_gt = len(matches)
    unused_pred_indices = np.array(sorted(list(available)), dtype=int)
    return matches, mapped_gt, unused_pred_indices
import numpy as np

def greedy_match_by_y_endpoints(
    pred_lines: np.ndarray,
    gt_lines: np.ndarray,
    max_y_mae: float | None = None,   # unused (kept for signature compatibility)
    print_first: int = 5
):
    """
    Greedy GT→Pred assignment using y-endpoint absolute differences.

    Behavior (per your spec):
      • Assumes caller already oriented lines left→right.
      • Sort both GT and Pred by start-point y (index [:,0,1]).
      • For each GT (in that sorted order), consider ALL remaining preds and
        pick the one with minimum cost:
            cost = |y0_pred - y0_gt| + |y1_pred - y1_gt|
      • No threshold: every GT gets the best currently-available pred.
      • Returns ORIGINAL indices for both GT and Pred.

    IMPORTANT: pred_lines and gt_lines must be in the same coordinate space.
    If your predictions are in 224×224, rescale them to GT space before calling.

    Args:
        pred_lines: (Np, 2, 2)  [(x0,y0),(x1,y1)] already left→right.
        gt_lines:   (Ng, 2, 2)  [(x0,y0),(x1,y1)] already left→right.
        max_y_mae:  kept only for API compatibility; ignored.
        print_first: how many example matches to print.

    Returns:
        matches: list of dicts {
            gt_idx, pred_idx, y_err0, y_err1, mae_y,
            gt_idx_sorted, pred_idx_sorted
        }   # gt_idx / pred_idx are ORIGINAL indices
        mapped_gt: int  (number of GTs that got a pred assigned)
        unused_pred_indices: np.ndarray of ORIGINAL pred indices not used
    """
    Ng = len(gt_lines)
    Np = len(pred_lines)

    if Ng == 0:
        return [], 0, np.arange(Np, dtype=int)
    if Np == 0:
        return [], 0, np.array([], dtype=int)

    # ---- sort both sets by start-point y (y at left endpoint)
    gt_sort_idx   = np.argsort(gt_lines[:, 0, 1])
    pred_sort_idx = np.argsort(pred_lines[:, 0, 1])

    gt_s   = gt_lines[gt_sort_idx]
    pred_s = pred_lines[pred_sort_idx]

    # y-only views: (start_y, end_y)
    gt_y   = np.stack([gt_s[:, 0, 1],   gt_s[:, 1, 1]], axis=1)   # (Ng, 2)
    pred_y = np.stack([pred_s[:, 0, 1], pred_s[:, 1, 1]], axis=1) # (Np, 2)

    # availability tracked in SORTED index space
    available = list(range(Np))
    matches = []

    for j_sorted in range(Ng):
        if not available:
            break

        ay = np.array(available, dtype=int)

        # Print current pool vs the GT under consideration (as requested)
        print("predicted points ", pred_y[ay], "ground truth points ", gt_y[j_sorted])

        # compute costs to all available preds for this GT
        diff = np.abs(pred_y[ay] - gt_y[j_sorted])  # (Na, 2)
        cost = diff.sum(axis=1)                     # (Na,)
        k = int(np.argmin(cost))
        p_sorted = int(ay[k])

        yerr0, yerr1 = float(diff[k, 0]), float(diff[k, 1])
        mae_y = 0.5 * (yerr0 + yerr1)

        # map back to ORIGINAL indices
        gt_orig   = int(gt_sort_idx[j_sorted])
        pred_orig = int(pred_sort_idx[p_sorted])

        matches.append({
            "gt_idx": gt_orig,
            "pred_idx": pred_orig,
            "y_err0": yerr0,
            "y_err1": yerr1,
            "mae_y": mae_y,
            "gt_idx_sorted": j_sorted,
            "pred_idx_sorted": p_sorted,
        })

        # remove chosen pred from the pool
        available.remove(p_sorted)

    # ---- Pretty print a few examples (using ORIGINAL indices)
    if print_first > 0 and matches:
        print(f"\n[greedy_match_by_y_endpoints] Showing first {min(print_first, len(matches))} matches:")
        for m in matches[:print_first]:
            print(
                f"  GT(orig:{m['gt_idx']:>3}, sort:{m['gt_idx_sorted']:>3})  "
                f"⇐  Pred(orig:{m['pred_idx']:>3}, sort:{m['pred_idx_sorted']:>3})  "
                f"| y_errs=({m['y_err0']:.2f}, {m['y_err1']:.2f})  mae_y={m['mae_y']:.2f}"
            )

    mapped_gt = len(matches)

    # remaining preds → ORIGINAL indices
    if len(available):
        unused_sorted = np.array(sorted(available), dtype=int)
        unused_pred_indices = pred_sort_idx[unused_sorted]
    else:
        unused_pred_indices = np.array([], dtype=int)

    return matches, mapped_gt, unused_pred_indices


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

    def rescale_lines_yx(lines, src_hw, dst_hw):
        lines = lines.astype(float).copy()
        sy = dst_hw[0] / float(src_hw[0]);
        sx = dst_hw[1] / float(src_hw[1])
        lines[:, :, 0] *= sx  # x
        lines[:, :, 1] *= sy  # y
        return lines

    pred_in_gt_space = rescale_lines_yx(filtered_lines, (224, 224), (128, 128))
    matches, mapped_gt, unused = greedy_match_by_y_endpoints(pred_in_gt_space, ground_truth_lines, print_first=5)

    # Y_MAE_THRESH = 3.0  # tweak as you like, or set to None
    #
    # matches, mapped_gt, unused_pred = greedy_match_by_y_endpoints(
    #     filtered_lines,
    #     ground_truth_lines,
    #     max_y_mae=Y_MAE_THRESH,
    #     print_first=5
    # )
    print(f"\nTotal GT mapped by y-endpoints: {mapped_gt} / {len(ground_truth_lines)}")
    print(f"Unused predictions left: {len(unused)}")


    tp, fp = msTPFP_sampels(filtered_lines, ground_truth_lines, threshold, debug=False)
    # tp, fp = msTPFP_endpt_mae_sequential(filtered_lines, ground_truth_lines,
    #                                      mae_thresh_px=threshold, debug=False)

    # tp, fp = msTPFP_endpt_mae_global(filtered_lines, ground_truth_lines,
    #                                  mae_thresh_px=threshold, debug=False, use_hungarian=False)

    num_gt = len(ground_truth_lines)
    print("TP:", tp.sum(), "FP:", fp.sum())
    return compute_ap(tp, fp,num_gt) * 100



# --- put this near the bottom of evaluation.py ---

from typing import Callable, Dict, List, Tuple

MatcherFn = Callable[[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray]]

def eval_sAP_dataset(
    dataset: List[Dict],
    thresholds=(5.0, 10.0, 15.0),
    matcher: MatcherFn = msTPFP_sampels,          # or msTPFP_oriented / msTPFP_endpt_mae_*
    use_masks: bool = True
) -> Dict[float, float]:
    """
    LCNN-style evaluation across a dataset.

    dataset: list of dict per image with keys:
      {
        "gt_lines":   (Ng, 2, 2) float32 in [x,y] (same coord frame across all images),
        "pred_lines": (Np, 2, 2) float32 in [x,y],
        "scores":     (Np,)      float32,
        "mask":       (H,W) uint8, 1=valid (optional; used only for pre-filtering)
      }

    thresholds: pixel tolerances (same units as your line coords).
    matcher:    returns (tp, fp) *in the SAME ORDER as pred_lines* for a given threshold.
    use_masks:  if True, we drop predicted lines whose endpoints fail the mask test you use elsewhere.
    """
    def passes_mask(line, mask, min_frac=0.6, samples=50):
        # Same logic as line_prediction.line_passes_mask (endpoints are [y,x] there),
        # but here our lines are [x,y], so we swap accordingly.
        if mask is None:
            return True
        (x0, y0), (x1, y1) = line
        H, W = mask.shape
        xs = np.linspace(x0, x1, samples)
        ys = np.linspace(y0, y1, samples)
        inside = valid = 0
        for xx, yy in zip(xs, ys):
            x = int(round(xx)); y = int(round(yy))
            if 0 <= y < H and 0 <= x < W:
                valid += 1
                inside += int(mask[y, x] > 0)
        return (valid > 0) and (inside / max(valid, 1)) >= min_frac

    # Collect GT size once for normalizing recall (sum over all images)
    n_gt_total = 0
    # For each threshold, we’ll accumulate TP/FP and Scores, then rank globally.
    out = {}
    for T in thresholds:
        all_tp = []
        all_fp = []
        all_scores = []
        n_gt_total = 0

        for item in dataset:
            gt = np.asarray(item["gt_lines"], dtype=float)
            pred = np.asarray(item["pred_lines"], dtype=float)
            s = np.asarray(item["scores"], dtype=float)
            mask = item.get("mask", None) if use_masks else None

            n_gt_total += len(gt)

            # (optional) pre-filter predictions by mask
            if mask is not None and len(pred) > 0:
                keep = [i for i, L in enumerate(pred) if passes_mask(L, mask)]
                if len(keep) == 0:
                    continue
                pred = pred[keep]
                s = s[keep]

            if len(pred) == 0 or len(gt) == 0:
                # no preds or no GT → all predictions are FP; add their scores so ranking is preserved
                all_tp.append(np.zeros(len(pred), float))
                all_fp.append(np.ones(len(pred), float))
                all_scores.append(s)
                continue

            # Compute TP/FP for this image at tolerance T (order = input order)
            tp_i, fp_i = matcher(pred, gt, T, debug=False)  # uses your existing matcher
            all_tp.append(tp_i)
            all_fp.append(fp_i)
            all_scores.append(s)

        # Concatenate over images and do the global sort by score (descending)
        if len(all_scores) == 0 or n_gt_total == 0:
            out[T] = 0.0
            continue

        tp = np.concatenate(all_tp) if len(all_tp) else np.zeros(0, float)
        fp = np.concatenate(all_fp) if len(all_fp) else np.zeros(0, float)
        scores = np.concatenate(all_scores) if len(all_scores) else np.zeros(0, float)

        order = np.argsort(-scores)
        tp = np.cumsum(tp[order]) / float(n_gt_total)
        fp = np.cumsum(fp[order]) / float(n_gt_total)

        # exactly the same AP as LCNN (monotone precision envelope)
        ap_T = compute_ap(tp, fp, num_gt=1)  # tp/fp are already normalized by n_gt_total
        out[T] = float(ap_T) * 100.0

    return out

import glob
from pathlib import Path

def load_gt_lines_from_label_npz(label_npz_path: str) -> np.ndarray:
    """
    Try to read GT lines from a *_label.npz produced by save_heatmap().
    If your NPZ uses a different key name, adapt here.
    """
    data = np.load(label_npz_path, allow_pickle=True)
    # Common patterns: 'lpos' (LCNN), or you may have stored 'lines'
    if "lines" in data.files:
        return data["lines"].astype(float).reshape(-1, 2, 2)
    if "lpos" in data.files:
        return data["lpos"].astype(float)[:, :, :2]  # (N,2,2) and drop extra dims if present
    raise KeyError(f"Could not find GT lines in {label_npz_path}. Available keys: {data.files}")

def build_dataset_from_folders(
    pred_npz_dir: str,       # e.g. "outputs/reals/pred_npz/post"
    gt_npz_dir: str,         # e.g. "outputs/reals/npz_data"
    use_masks=True
) -> List[Dict]:
    """
    Match prediction NPZs to GT/mask NPZs by shared stem.
    Returns a list of dict items ready for eval_sAP_dataset().
    """
    pred_files = sorted(glob.glob(str(Path(pred_npz_dir) / "*.npz")))
    items = []
    for pf in pred_files:
        stem = Path(pf).stem   # matches JSON stem you saved earlier
        # prediction
        P = np.load(pf)
        pred_lines = P["lines"].astype(float)  # [x,y]
        scores = P["score"].astype(float)

        # ground truth
        label_npz = str(Path(gt_npz_dir) / f"{stem}_label.npz")
        if not Path(label_npz).exists():
            # fall back: maybe your saved GT used the pre-crop stem (before suffixes)
            # adapt this if your stems differ.
            continue
        gt_lines = load_gt_lines_from_label_npz(label_npz)

        # optional mask
        mask_npz = str(Path(gt_npz_dir) / f"{stem}_mask_label.npz")
        mask = None
        if use_masks and Path(mask_npz).exists():
            M = np.load(mask_npz)
            mask = M["mask"].astype(np.uint8)

        items.append({
            "gt_lines": gt_lines,
            "pred_lines": pred_lines,
            "scores": scores,
            "mask": mask
        })
    return items

def eval_sAP_from_folders(
    pred_npz_dir: str,
    gt_npz_dir: str,
    thresholds=(5.0,10.0,15.0),
    matcher: MatcherFn = msTPFP_sampels,
    use_masks=True
):
    dataset = build_dataset_from_folders(pred_npz_dir, gt_npz_dir, use_masks=use_masks)
    return eval_sAP_dataset(dataset, thresholds=thresholds, matcher=matcher, use_masks=use_masks)
