import numpy as np
import matplotlib.pyplot as plt
import os
import glob

#
# GT = "data/wireframe/valid/*.npz"

# GT = "data/pcwireframe_test/test/*.npz"

#  python eval-sAP.py results_ct5k1
# python eval-sAP.py results_all
# python eval-sAP.py results_clst5knew
# GT = "data/pcwireframe_clst5knew/test/*.npz"

# GT = "data/pcwireframe_ct5k1/test/*.npz"


#  python eval-sAP.py results_ct5kde1
# GT = "data/pcwireframe_ct5kde1/test/*.npz"

#python eval-sAP.py results_clst5kdenew
# GT = "data/pcwireframe_clst5kdenew/test/*.npz"

#
# def line_score(path, threshold=5):
#     preds = sorted(glob.glob(path))
#     gts = sorted(glob.glob(GT))
#
#     n_gt = 0
#     lcnn_tp, lcnn_fp, lcnn_scores = [], [], []
#     for pred_name, gt_name in zip(preds, gts):
#         with np.load(pred_name) as fpred:
#             lcnn_line = fpred["lines"][:, :, :2]
#             lcnn_score = fpred["score"]
#         with np.load(gt_name) as fgt:
#             gt_line = fgt["lpos"][:, :, :2]
#         n_gt += len(gt_line)
#
#         for i in range(len(lcnn_line)):
#             if i > 0 and (lcnn_line[i] == lcnn_line[0]).all():
#                 lcnn_line = lcnn_line[:i]
#                 lcnn_score = lcnn_score[:i]
#                 break
#
#         tp, fp = lcnn.metric.msTPFP(lcnn_line, gt_line, threshold)
#         lcnn_tp.append(tp)
#         lcnn_fp.append(fp)
#         lcnn_scores.append(lcnn_score)
#
#     lcnn_tp = np.concatenate(lcnn_tp)
#     lcnn_fp = np.concatenate(lcnn_fp)
#     lcnn_scores = np.concatenate(lcnn_scores)
#     lcnn_index = np.argsort(-lcnn_scores)
#     lcnn_tp = np.cumsum(lcnn_tp[lcnn_index]) / n_gt
#     lcnn_fp = np.cumsum(lcnn_fp[lcnn_index]) / n_gt
#
#     return lcnn.metric.ap(lcnn_tp, lcnn_fp)
#
#
#
#
# if __name__ == "__main__":
#     args = docopt(__doc__)
#
#     def work(path):
#         print(f"Working on {path}")
#         return [100 * line_score(f"{path}/*.npz", t) for t in [5, 10, 15]]
#
#     dirs = sorted(sum([glob.glob(p) for p in args["<path>"]], []))
#     results = lcnn.utils.parmap(work, dirs)
#
#     for d, msAP in zip(dirs, results):
#         print(f"{d}: {msAP[0]:2.1f} {msAP[1]:2.1f} {msAP[2]:2.1f}")
#

def process_line_detection(gt_path, pred_path, mask_path, threshold=10, eps=1.0):
    """
    Processes line detection by loading ground truth, predictions, and masks.
    Filters and evaluates predicted lines using mAP (mean Average Precision).

    Args:
        gt_path (str): Path to the ground truth .npz file.
        pred_path (str): Path to the predicted lines .npz file.
        mask_path (str): Path to the mask .npz file.
        threshold (float): Distance threshold for line matching.
        eps (float): Distance threshold for removing duplicate lines.

    Returns:
        float: mAP score for the file.
    """
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
        return map_score * 100  # Convert to percentage
    except Exception as e:
        print(f"Error processing {gt_path}: {e}")
        return None


def process_multiple_files(gt_dir, pred_dir, mask_dir):
    """
    Processes multiple files in the specified directories and computes the average mAP.

    Args:
        gt_dir (str): Directory containing ground truth .npz files.
        pred_dir (str): Directory containing predicted lines .npz files.
        mask_dir (str): Directory containing mask .npz files.

    Returns:
        float: Average mAP score over all files.
    """
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.npz")))
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.npz")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.npz")))

    if not (gt_files and pred_files and mask_files):
        print("Error: Missing files in one or more directories.")
        return None

    total_map = 0
    valid_count = 0

    for gt_file, pred_file, mask_file in zip(gt_files, pred_files, mask_files):
        map_score = process_line_detection(gt_file, pred_file, mask_file)
        if map_score is not None:
            total_map += map_score
            valid_count += 1
            print(f"Processed: {gt_file} | mAP: {map_score:.4f}")

    avg_map = total_map / valid_count if valid_count > 0 else 0
    print(f"\nTotal Processed Files: {valid_count}")
    print(f"Average mAP Score: {avg_map:.4f}")
    return avg_map


# Example Usage
gt_dir = "clst/"
pred_dir = "clst/clst_pred/"
mask_dir = "clst/mask/"

average_map = process_multiple_files(gt_dir, pred_dir, mask_dir)
