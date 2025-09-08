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


def msTPFP1(line_pred, line_gt, threshold):
    """Compute true positives (TP) and false positives (FP) for line matching."""
    diff = ((line_pred[:, None, :, None] - line_gt[:, None]) ** 2).sum(-1)
    diff = np.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1],
        diff[:, :, 0, 1] + diff[:, :, 1, 0]
    )
    choice = np.argmin(diff, 1)
    dist = np.min(diff, 1)

    hit = np.zeros(len(line_gt), bool)
    tp = np.zeros(len(line_pred), float)
    fp = np.zeros(len(line_pred), float)

    for i in range(len(line_pred)):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1
    return tp, fp


def compute_ap(tp, fp):
    """Compute Average Precision (AP) from TP/FP arrays."""
    recall = np.cumsum(tp) / max(len(tp), 1)
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-9)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    i = np.where(recall[1:] != recall[:-1])[0]
    return np.sum((recall[i + 1] - recall[i]) * precision[i + 1])

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

    tp, fp = msTPFP(filtered_lines, ground_truth_lines, threshold, debug=False)
    print("TP:", tp.sum(), "FP:", fp.sum())
    return compute_ap(tp, fp) * 100




