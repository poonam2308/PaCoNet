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
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt

import src.dhlp.lcnn.utils
import src.dhlp.lcnn.metric

# GT = "data/pcwireframe_test/test/*.npz"
# MASK_PATH = "data/pcwireframe_test/masks/*.npz"

# python eval-sAP.py results_scat
# GT = "data/pcwireframe_scat/test/*.npz"
# MASK_PATH = "data/pcwireframe_scat/masks/*.npz"

#color
#
# GT = "data/pcwireframe_ct5k1/test/*.npz"
# MASK_PATH = "data/pcwireframe_ct5k1/masks/*.npz"

# GT = "data/pcwireframe_ct5kde1/test/*.npz"
# MASK_PATH = "data/pcwireframe_ct5kde1/masks/*.npz"

# cluster
#
# GT = "data/pcwireframe_clst5knew/test/*.npz"
# MASK_PATH = "data/pcwireframe_clst5knew/masks/*.npz"

GT = "data/pcwireframe_clst5kdenew/test/*.npz"
MASK_PATH = "data/pcwireframe_clst5kdenew/masks/*.npz"


def line_score(pred_path, threshold=1):
    pred_files = sorted(glob.glob(pred_path))  # Load predicted npz files
    gt_files = sorted(glob.glob(GT))  # Load ground truth npz files

    # Create dictionaries for filename-based matching
    pred_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in pred_files}
    gt_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in gt_files}

    # Find common filenames
    common_files = sorted(set(pred_dict.keys()) & set(gt_dict.keys()))

    if not common_files:
        print("No matching files found between predictions and ground truth.")
        return 0  # Return 0 if no matches

    n_gt = 0
    lcnn_tp, lcnn_fp, lcnn_scores = [], [], []

    for filename in common_files:
        pred_name = pred_dict[filename]
        gt_name = gt_dict[filename]

        with np.load(pred_name) as fpred:
            lcnn_line = fpred["lines"][:, :, :2]
            lcnn_score = fpred["score"]

        with np.load(gt_name) as fgt:
            gt_line = fgt["lpos"][:, :, :2]

        n_gt += len(gt_line)

        # Avoid duplicate predictions
        unique_lines = []
        unique_scores = []
        for i in range(len(lcnn_line)):
            if i > 0 and (lcnn_line[i] == lcnn_line[0]).all():
                break
            unique_lines.append(lcnn_line[i])
            unique_scores.append(lcnn_score[i])

        unique_lines = np.array(unique_lines)
        unique_scores = np.array(unique_scores)

        tp, fp = lcnn.metric.msTPFP(unique_lines, gt_line, threshold)
        #plot_data(unique_lines, gt_line, threshold, tp,fp)
        lcnn_tp.append(tp)
        lcnn_fp.append(fp)
        lcnn_scores.append(unique_scores)

    if n_gt == 0:
        print("Warning: No ground truth lines found.")
        return 0  # Avoid division by zero

    lcnn_tp = np.concatenate(lcnn_tp) if lcnn_tp else np.array([])
    lcnn_fp = np.concatenate(lcnn_fp) if lcnn_fp else np.array([])
    lcnn_scores = np.concatenate(lcnn_scores) if lcnn_scores else np.array([])

    if lcnn_scores.size > 0:
        lcnn_index = np.argsort(-lcnn_scores)
        lcnn_tp = np.cumsum(lcnn_tp[lcnn_index]) / n_gt
        lcnn_fp = np.cumsum(lcnn_fp[lcnn_index]) / n_gt
        return lcnn.metric.ap(lcnn_tp, lcnn_fp)

    return 0  # No valid predictions


def plot_data(pred_lines_example, gt_lines_example, threshold_example, tp_example, fp_example):
    """Plots three figures: (1) Combined GT + Predicted, (2) GT Only, (3) Predicted Only"""

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # --- 1. Plot Ground Truth and Predicted Lines Together ---
    ax = axs[0]
    ax.set_title("GT & Predicted (TP in Green, FP in Red)")

    # Plot GT lines (blue)
    for line in gt_lines_example:
        ax.plot([line[0][1], line[1][1]], [line[0][0], line[1][0]], 'bo-', label="GT Line")

    # Plot Predicted lines (Green for TP, Red for FP)
    for i, line in enumerate(pred_lines_example):
        color = 'g' if tp_example[i] == 1 else 'r'
        ax.plot([line[0][1], line[1][1]], [line[0][0], line[1][0]], color + '-', linewidth=2)

    ax.set_xlim(0, 150)
    ax.set_ylim(0, 150)
    ax.invert_yaxis()

    # --- 2. Plot Ground Truth Only ---
    ax = axs[1]
    ax.set_title("Ground Truth Lines (GT)")

    for line in gt_lines_example:
        ax.plot([line[0][1], line[1][1]], [line[0][0], line[1][0]], 'bo-', label="GT Line")

    ax.set_xlim(0, 150)
    ax.set_ylim(0, 150)
    ax.invert_yaxis()

    # --- 3. Plot Predicted Only (TP in Green, FP in Red) ---
    ax = axs[2]
    ax.set_title("Predicted Lines (TP in Green, FP in Red)")

    for i, line in enumerate(pred_lines_example):
        color = 'g' if tp_example[i] == 1 else 'r'
        ax.plot([line[0][1], line[1][1]], [line[0][0], line[1][0]], color + '-', linewidth=2)

    ax.set_xlim(0, 150)
    ax.set_ylim(0, 150)
    ax.invert_yaxis()

    plt.show()


def process_line_detection(gt_path, pred_path, mask_path, threshold=1, eps=5):
    try:
        # Load predicted lines and scores
        predictions_data = np.load(pred_path)
        predicted_lines = predictions_data["lines"]  # Shape: (N, 2, 2)

        # Load the binary mask
        mask_data = np.load(mask_path)
        mask = mask_data["mask"]  # Binary mask of shape (h, w)

        # Load ground truth lines
        gt_data = np.load(gt_path)
        ground_truth_lines = gt_data["lpos"][:, :, :2]  # Extract (x, y) coordinates

        # Ensure lines are oriented from left to right
        def ensure_left_to_right(lines):
            corrected_lines = []
            for line in lines:
                (x0, y0), (x1, y1) = line
                if x0 > x1:  # Swap if not left to right
                    line = [[x1, y1], [x0, y0]]
                corrected_lines.append(line)
            return np.array(corrected_lines)

        predicted_lines = ensure_left_to_right(predicted_lines)
        ground_truth_lines = ensure_left_to_right(ground_truth_lines)

        # Function to filter lines based on mask
        def filter_lines_with_mask(lines, mask):
            valid_lines = []
            h, w = mask.shape
            for line in lines:
                (a, b) = line.astype(int)
                y0, x0 = a[0], a[1]
                y1, x1 = b[0], b[1]

                if 0 <= y0 < h and 0 <= x0 < w and 0 <= y1 < h and 0 <= x1 < w:
                    if mask[y0, x0] == 1 and mask[y1, x1] == 1:
                        valid_lines.append(line)
            return np.array(valid_lines)

        filtered_lines = filter_lines_with_mask(predicted_lines, mask)

        # Remove duplicate or near-identical lines
        def remove_duplicate_lines(lines, eps=1.0):
            unique_lines = []
            for line in lines:
                if all(np.linalg.norm(line - ul) > eps for ul in unique_lines):
                    unique_lines.append(line)
            return np.array(unique_lines)

        filtered_lines = remove_duplicate_lines(filtered_lines, eps)

        # Compute mAP for filtered lines
        def msTPFP(line_pred, line_gt, threshold):
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
            recall = np.cumsum(tp) / max(len(tp), 1)
            precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-9)

            recall = np.concatenate(([0.0], recall, [1.0]))
            precision = np.concatenate(([0.0], precision, [0.0]))

            for i in range(len(precision) - 1, 0, -1):
                precision[i - 1] = max(precision[i - 1], precision[i])

            i = np.where(recall[1:] != recall[:-1])[0]
            return np.sum((recall[i + 1] - recall[i]) * precision[i + 1])

        def compute_map(filtered_lines, ground_truth_lines, threshold=10):
            if len(filtered_lines) == 0 or len(ground_truth_lines) == 0:
                return 0
            tp, fp = msTPFP(filtered_lines, ground_truth_lines, threshold)
            return compute_ap(tp, fp)

        map_score = compute_map(filtered_lines, ground_truth_lines, threshold)
        return map_score * 100, filtered_lines.tolist(), ground_truth_lines.tolist() # Convert to percentage
    except Exception as e:
        print(f"Error processing {gt_path}: {e}")
        return None


def process_multiple_files(pred_dir, threshold, output_json_path):

    gt_files = sorted(glob.glob(GT))
    pred_files = sorted(glob.glob(pred_dir))
    mask_files = sorted(glob.glob(MASK_PATH))

    if not (gt_files and pred_files and mask_files):
        print("Error: Missing files in one or more directories.")
        return None

    # Extract filenames without extensions
    gt_filenames = {os.path.splitext(os.path.basename(f))[0]: f for f in gt_files}
    pred_filenames = {os.path.splitext(os.path.basename(f))[0]: f for f in pred_files}
    mask_filenames = {os.path.splitext(os.path.basename(f))[0]: f for f in mask_files}

    # Find common filenames across all three directories
    common_filenames = set(gt_filenames.keys()) & set(pred_filenames.keys()) & set(mask_filenames.keys())

    if not common_filenames:
        print("Error: No matching files found across all directories.")
        return None

    total_map = 0
    valid_count = 0
    results ={}
    for filename in sorted(common_filenames):
        gt_file = gt_filenames[filename]
        pred_file = pred_filenames[filename]
        mask_file = mask_filenames[filename]

        map_score, filtered_pred_lines, gt_lines = process_line_detection(gt_file,
                                                                          pred_file,
                                                                          mask_file,
                                                                          threshold)
        if filtered_pred_lines is not None and gt_lines is not None:
            results[filename] = {
                "filtered_predicted_lines": filtered_pred_lines,
                "ground_truth_lines": gt_lines,
            }
        if map_score is not None:
            total_map += map_score
            valid_count += 1
            # print(f"Processed: {filename} | mAP: {map_score:.4f}")
            #print(f"Processed: {filename} | mAP (t={threshold}): {map_score:.4f}")
    # Save results to JSON file
    with open(output_json_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Results saved to {output_json_path}")
    avg_map = total_map / valid_count if valid_count > 0 else 0
    print(f"\nTotal Processed Files: {valid_count}")
    # print(f"Average mAP Score: {avg_map:.4f}")
    print(f"Average mAP Score for t={threshold}: {avg_map:.4f}")
    return avg_map

if __name__ == "__main__":
    args = docopt(__doc__)


    def work(path):
        print(f"Working on {path}")
        for t in [5,10,15]:
            print(f"\nRunning process_multiple_files for threshold t={t}\n")
            avg_map = 100 * process_multiple_files(f"{path}/*.npz", t,
                                                   "output_json_data/results_clst5kdenew1_2.json")

        # return 100 * process_multiple_files(f"{path}/*.npz")
        # return [100 * line_score(f"{path}/*.npz", t) for t in [5, 10, 15]]


    dirs = sorted(sum([glob.glob(p) for p in args["<path>"]], []))
    results = lcnn.utils.parmap(work, dirs)

    for d, msAP in zip(dirs, results):
        print(f"{d}: {msAP[0]:2.1f} {msAP[1]:2.1f} {msAP[2]:2.1f}")

# results_ct5k1 54.2 54.5 54.6
# results_ct5kde1: 61.3 62.0 62.3
# results_clst5kdenew: 57.1 58.5 58.9
# results_all: 42.8 42.9 42.9

