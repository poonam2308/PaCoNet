import os
import random
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch



import numpy as np

def wasserstein_1d(u, v):
    """Exact 1D Wasserstein-1 for uniform weights."""
    u = np.sort(u)
    v = np.sort(v)
    n = len(u); m = len(v)
    # resample to same length by interpolation
    k = max(n, m)
    uq = np.interp(np.linspace(0, 1, k), np.linspace(0, 1, n), u)
    vq = np.interp(np.linspace(0, 1, k), np.linspace(0, 1, m), v)
    return float(np.mean(np.abs(uq - vq)))

def sliced_wasserstein_2d(pred_points, gt_points, n_proj=128, seed=0):
    P = np.asarray(pred_points, np.float64)
    G = np.asarray(gt_points, np.float64)
    if len(P) == 0 or len(G) == 0:
        return np.nan

    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_proj, 2))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    vals = []
    for d in dirs:
        p1 = P @ d
        g1 = G @ d
        vals.append(wasserstein_1d(p1, g1))
    return float(np.mean(vals))

def nearest_junction(point, junctions):
    distances = np.linalg.norm(junctions - point, axis=1)
    return junctions[np.argmin(distances)]  # Closest ground-truth junction

def compute_line_stats(lines):

    lines = np.array(lines)  # Convert to numpy array for easier computation

    start_points = lines[:, :2]  # Extract start points (x_s, y_s)
    end_points = lines[:, 2:]  # Extract end points (x_e, y_e)

    # Compute means
    mean_start = np.mean(start_points, axis=0)
    mean_end = np.mean(end_points, axis=0)

    # Compute standard deviations
    std_start = np.std(start_points, axis=0)
    std_end = np.std(end_points, axis=0)

    # Compute upper and lower bounds (1 standard deviation above/below the mean)
    lower_start = mean_start - std_start
    upper_start = mean_start + std_start
    lower_end = mean_end - std_end
    upper_end = mean_end + std_end

    return {
        "mean_start": tuple(mean_start),
        "mean_end": tuple(mean_end),
        "std_start": tuple(std_start),
        "std_end": tuple(std_end),
        "lower_start": tuple(lower_start),
        "upper_start": tuple(upper_start),
        "lower_end": tuple(lower_end),
        "upper_end": tuple(upper_end),
    }


def compute_distribution_stats(points):
    """
    points: (N, 2) array of [x, y]
    """
    points = np.asarray(points)
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    return mean, std

def compute_distribution_mae(pred_points, gt_points):
    """
    Compare distribution of predictions vs GT
    """
    mu_pred, std_pred = compute_distribution_stats(pred_points)
    mu_gt, std_gt = compute_distribution_stats(gt_points)

    mae_mean = np.mean(np.abs(mu_pred - mu_gt))
    mae_std = np.mean(np.abs(std_pred - std_gt))

    return {
        "pred_mean": mu_pred,
        "gt_mean": mu_gt,
        "pred_std": std_pred,
        "gt_std": std_gt,
        "mae_mean": mae_mean,
        "mae_std": mae_std,
    }
def compute_median_mad(points):
    median = np.median(points, axis=0)
    mad = np.median(np.abs(points - median), axis=0)
    return median, mad

def compute_distribution_mae_median(pred_points, gt_points):
    """
    Compare distribution of predictions vs GT
    """
    med_pred, mad_pred = compute_median_mad(pred_points)
    med_gt, mad_gt = compute_median_mad(gt_points)

    center_err = np.mean(np.abs(med_pred - med_gt))
    spread_err = np.mean(np.abs(mad_pred - mad_gt))

    return {
        "med_pred": med_pred,
        "med_gt": med_gt,
        "mad_pred": mad_pred,
        "mad_gt": mad_gt,
        "mae_center": center_err,
        "mae_spread_err": spread_err,
    }

import ot

def wasserstein_2d(pred_points, gt_points):
    P = np.asarray(pred_points, np.float64)
    G = np.asarray(gt_points, np.float64)
    if len(P) == 0 or len(G) == 0:
        return np.nan

    a = np.ones(len(P), dtype=np.float64) / len(P)
    b = np.ones(len(G), dtype=np.float64) / len(G)

    C = ot.dist(P, G, metric="euclidean")  # (N,M), in pixels
    return float(ot.emd2(a, b, C))


def chamfer_distance_2d(pred_points, gt_points, squared=False):
    """
    Symmetric Chamfer distance between two 2D point sets.

    pred_points: (N,2)
    gt_points:   (M,2)

    Returns a single scalar in pixels (or pixels^2 if squared=True).
    """
    pred_points = np.asarray(pred_points, dtype=np.float32)
    gt_points   = np.asarray(gt_points, dtype=np.float32)

    if len(pred_points) == 0 or len(gt_points) == 0:
        return np.nan  # or return a large number / 0 depending on your preference

    # pairwise squared distances: (N,M)
    diff = pred_points[:, None, :] - gt_points[None, :, :]
    d2 = np.sum(diff * diff, axis=2)

    if squared:
        d_pred_to_gt = np.min(d2, axis=1)  # (N,)
        d_gt_to_pred = np.min(d2, axis=0)  # (M,)
    else:
        d_pred_to_gt = np.sqrt(np.min(d2, axis=1))
        d_gt_to_pred = np.sqrt(np.min(d2, axis=0))

    # symmetric chamfer: mean both directions
    return float(np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred))



def compute_average_offset_errors(offset_errors_list):
    mean_offsets = [e['mean_offset_error'] for e in offset_errors_list]
    lower_offsets = [e['lower_offset_error'] for e in offset_errors_list]
    upper_offsets = [e['upper_offset_error'] for e in offset_errors_list]
    overall_offsets = [e['overall_offset_error'] for e in offset_errors_list]

    avg_mean_offset = np.mean(mean_offsets) if mean_offsets else 0
    avg_lower_offset = np.mean(lower_offsets) if lower_offsets else 0
    avg_upper_offset = np.mean(upper_offsets) if upper_offsets else 0
    avg_overall_offset = np.mean(overall_offsets) if overall_offsets else 0

    return {
        "avg_mean_offset": avg_mean_offset,
        "avg_lower_offset": avg_lower_offset,
        "avg_upper_offset": avg_upper_offset,
        "avg_overall_offset": avg_overall_offset
    }


def compute_offset_error(pred_lines, gt_lines):

    def compute_stats(lines):
        """Helper function to compute mean, std, lower and upper bounds."""
        lines = np.array(lines)
        mean = np.mean(lines, axis=0)
        std = np.std(lines, axis=0)
        lower = mean - std
        upper = mean + std
        return mean, std, lower, upper

    # Compute stats for predicted and ground truth lines
    mean_pred, std_pred, lower_pred, upper_pred = compute_stats(pred_lines)
    mean_gt, std_gt, lower_gt, upper_gt = compute_stats(gt_lines)

    # Compute offset errors
    mean_offset_error = np.linalg.norm(mean_pred - mean_gt)
    lower_offset_error = np.linalg.norm(lower_pred - lower_gt)
    upper_offset_error = np.linalg.norm(upper_pred - upper_gt)

    # Compute overall offset error
    overall_offset_error = (mean_offset_error + lower_offset_error + upper_offset_error) / 3

    return {
        "mean_offset_error": mean_offset_error,
        "lower_offset_error": lower_offset_error,
        "upper_offset_error": upper_offset_error,
        "overall_offset_error": overall_offset_error
    }

def compute_nearest_junction_offset_stats(pred_lines, gt_junctions):
    start_offsets = []
    end_offsets = []

    for line in pred_lines:
        start, end = line[:2], line[2:]

        nearest_start = nearest_junction(start, gt_junctions)
        nearest_end = nearest_junction(end, gt_junctions)
        start_offsets.append(np.linalg.norm(np.array(start) - np.array(nearest_start)))
        end_offsets.append(np.linalg.norm(np.array(end) - np.array(nearest_end)))

    all_offsets = np.array(start_offsets + end_offsets)
    cap = 30.0  # pixels
    all_offsets = np.minimum(all_offsets, cap)
    mean_offset = np.mean(all_offsets) if len(all_offsets) > 0 else 0
    std_offset = np.std(all_offsets) if len(all_offsets) > 0 else 0
    lower_bound = mean_offset - std_offset
    upper_bound = mean_offset + std_offset

    return {
        "mean_offset": mean_offset,
        "std_offset": std_offset,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }


def nearest_junction_better(point, junctions):
    """
    Finds the nearest junction to a given point.

    Args:
        point (array-like): Coordinates of the point [x, y].
        junctions (array-like): Array of junction coordinates [[x1, y1], [x2, y2], ...].

    Returns:
        array: Coordinates of the nearest junction [x, y].
        float: Distance to the nearest junction.
    """
    # Convert inputs to numpy arrays for safety
    point = np.array(point)
    junctions = np.array(junctions)

    # Handle empty junctions
    if len(junctions) == 0:
        raise ValueError("No junctions available to find the nearest one.")

    # Compute distances
    distances = np.linalg.norm(junctions - point, axis=1)

    # Find the nearest junction
    nearest_idx = np.argmin(distances)
    nearest_junction = junctions[nearest_idx]
    nearest_distance = distances[nearest_idx]

    return nearest_junction, nearest_distance

def line_angle(line):
    """Calculate the angle of a line in radians."""
    (x1, y1), (x2, y2) = line  # Adjusted unpacking
    return np.arctan2(y2 - y1, x2 - x1)

def line_overlap(line1, line2):
    """Calculate the overlap between two collinear lines."""
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    if abs(line_angle(line1) - line_angle(line2)) > 1e-2:  # Check if angles are similar
        return 0
    start = max(min(x1, x2), min(x3, x4))
    end = min(max(x1, x2), max(x3, x4))
    overlap = max(0, end - start)
    return overlap / max(abs(x2 - x1), abs(x4 - x3))  # Normalize by line length

def nms_lines_sim(lines, scores, threshold=0.5):
    """
    Perform Non-Maximum Suppression on lines.
    Args:
        lines: List of lines, each represented as [(x1, y1), (x2, y2)].
        scores: List of confidence scores for each line.
        threshold: Suppression threshold for similarity (0 to 1).
    Returns:
        List of retained lines after NMS.
    """
    indices = np.argsort(scores)[::-1]  # Sort by score (descending)
    retained_lines = []

    while len(indices) > 0:
        best_idx = indices[0]
        retained_lines.append(lines[best_idx])
        rest = indices[1:]

        suppressed = []
        for i in rest:
            similarity = line_overlap(lines[best_idx], lines[i])
            if similarity < threshold:
                suppressed.append(i)

        indices = np.array(suppressed)

    return np.array(retained_lines)

## non maximum supression

def nms_lines(lines, scores, threshold=70):
    """Perform Non-Maximum Suppression (NMS) to remove overlapping lines."""
    indices = np.argsort(-scores)  # Sort by confidence score (descending)
    selected_lines = []
    selected_scores = []

    for idx in indices:
        line = lines[idx]
        keep = True
        for sel_line in selected_lines:
            dist = np.linalg.norm(line - sel_line)  # Measure distance
            if dist < threshold:
                keep = False
                break
        if keep:
            selected_lines.append(line)
            selected_scores.append(scores[idx])

    return np.array(selected_lines), np.array(selected_scores)


def filter_lines_by_junctions(lines, junctions, threshold=5):
    """Remove lines that do not connect high-confidence junctions."""
    filtered_lines = []
    for (a, b) in lines:
        dist_a = np.linalg.norm(junctions - a, axis=1)
        dist_b = np.linalg.norm(junctions - b, axis=1)

        if np.min(dist_a) < threshold and np.min(dist_b) < threshold:
            filtered_lines.append((a, b))

    return np.array(filtered_lines)


def calculate_filtered_line_alignment(mask, filtered_lines):
    """
    Calculate the probability that filtered predicted lines align over the valid mask region.

    Args:
        mask (torch.Tensor): Binary mask (1 = valid, 0 = masked/white).
        filtered_lines (np.array): Filtered predicted lines [(x1, y1, x2, y2), ...]

    Returns:
        float: Probability of line alignment.
    """
    # Convert mask to NumPy array
    mask_np = mask.squeeze(0).numpy()  # Shape (H, W)

    # Create a blank image of the same size as the mask
    h, w = mask_np.shape
    line_canvas = np.zeros((h, w), dtype=np.uint8)

    # Draw the filtered lines on the blank canvas
    for (a, b) in filtered_lines:
        [a[1], b[1]], [a[0], b[0]]
        cv2.line(line_canvas,  [int(a[1]), int(b[1])], [int(a[0]), int(b[0])], color=1, thickness=1)

    # Compute overlap between drawn lines and valid mask regions
    line_pixels = np.sum(line_canvas)  # Total line pixels drawn
    valid_line_pixels = np.sum(line_canvas * mask_np)  # Only count pixels in valid areas

    # Compute probability of alignment
    if line_pixels == 0:
        return 0.0  # Avoid division by zero

    alignment_prob = valid_line_pixels / line_pixels
    return alignment_prob


def generate_mask(image_path, white_thresh=250):

    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Create binary mask: White pixels -> 0 (mask), Non-white pixels -> 1 (valid)
    mask = (image_gray < white_thresh).astype(np.uint8)

    # Convert to PyTorch tensor
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Shape: (1, H, W)
    return mask_tensor



def visualize_points(img_tensor,
                     gt_junc,
                     start_points,
                     end_points,
                     save_path,
                     title=None):
    img = img_tensor.cpu().numpy()
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))  # C,H,W -> H,W,C

    # Normalize image for display
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img_vis = (img - img_min) / (img_max - img_min)
    else:
        img_vis = img

    plt.figure(figsize=(6, 6))
    plt.imshow(img_vis)

    # GT junctions (black circles, open)
    if gt_junc is not None and gt_junc.size > 0:
        gt_pts = gt_junc[:, :2]
        plt.scatter(gt_pts[:, 1], gt_pts[:, 0],
                    s=10, marker='o', edgecolors='k', facecolors='none',
                    label='GT junction')

    # Predicted start points (green x)
    if start_points is not None and start_points.size > 0:
        plt.scatter(start_points[:, 1], start_points[:, 0],
                    s=10, marker='x', color='g', label='Pred start')

    # Predicted end points (blue +)
    if end_points is not None and end_points.size > 0:
        plt.scatter(end_points[:, 1], end_points[:, 0],
                    s=10, marker='+', color='b', label='Pred end')

    if title is not None:
        plt.title(title)

    plt.legend(loc='upper right', fontsize=8)
    plt.axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


#-------------------- masking and filtering -------------------

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_mask_for_index(dataset, index, MASK_ROOT , mask_key="mask"):
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
# Crop grouping
# ------------------------------------------------------------
def crop_key_from_label_path(label_path: str) -> str:
    """
    label path basename example:
      image_1_crop_1_BKxJsl_0_label.npz
    crop key:
      image_1_crop_1
    """
    base = os.path.basename(label_path).replace("_label.npz", "")
    parts = base.split("_")
    # expected: ["image","1","crop","1", "<catHash>", "0"]
    if len(parts) >= 4:
        return "_".join(parts[:4])
    return base

# ------------------------------------------------------------
# Mask utilities
# ------------------------------------------------------------
def get_mask_for_label_path(label_path: str, MASK_ROOT, mask_key="mask", ) -> torch.Tensor:
    """
    Uses MASK_ROOT + same basename as label npz.
    """
    base_name = os.path.basename(label_path)
    mask_path = os.path.join(MASK_ROOT, base_name)

    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Mask npz not found: {mask_path}")

    npz = np.load(mask_path)
    if mask_key not in npz.files:
        raise KeyError(f"Key '{mask_key}' not found in {mask_path}. Available: {npz.files}")

    mask = npz[mask_key]
    npz.close()

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    return torch.from_numpy(mask > 0)  # [H,W] bool


def coverage_mae_junctions(
    gt_juncs: np.ndarray,          # shape (J, 2)
    pred_points: np.ndarray,       # shape (K, 2) endpoints (start+end)
    miss_penalty: float = 30.0,    # pixels
) -> float:
    """
    Global-style MAE: average distance from each GT junction to nearest predicted point.
    If no predicted points exist, each GT junction gets miss_penalty.
    """
    if gt_juncs.size == 0:
        return 0.0
    if pred_points.size == 0:
        return float(miss_penalty)

    # For each GT junction, compute distance to nearest predicted point
    # (vectorized)
    diff = gt_juncs[:, None, :] - pred_points[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)  # (J, K)
    min_d = dists.min(axis=1)              # (J,)
    min_d = np.clip(min_d, 0, miss_penalty)
    return float(min_d.mean())


def _canon_lines_yx(lines: np.ndarray) -> np.ndarray:
    """lines: (n,2,2) in (y,x). Canonicalize so endpoint0 is 'leftmost' by x, tie by y."""
    if lines.size == 0:
        return lines.astype(np.float32)

    a = lines[:, 0, :]  # (n,2) (y,x)
    b = lines[:, 1, :]
    # compare by x then y
    swap = (b[:, 1] < a[:, 1]) | ((b[:, 1] == a[:, 1]) & (b[:, 0] < a[:, 0]))
    out = lines.copy().astype(np.float32)
    out[swap, 0, :] = b[swap]
    out[swap, 1, :] = a[swap]
    return out

def match_count_lines(gt_lines_yx: np.ndarray, pr_lines_yx: np.ndarray, max_cost: float = 10.0) -> int:
    """
    Returns m = number of matched lines (Hungarian + threshold).
    Cost = L2(start-start) + L2(end-end) after canonicalization.
    """
    if gt_lines_yx.size == 0 or pr_lines_yx.size == 0:
        return 0

    gt = _canon_lines_yx(gt_lines_yx)
    pr = _canon_lines_yx(pr_lines_yx)

    k = gt.shape[0]
    l = pr.shape[0]
    cost = np.zeros((k, l), dtype=np.float32)

    for i in range(k):
        gy0, gx0 = gt[i, 0]
        gy1, gx1 = gt[i, 1]
        for j in range(l):
            py0, px0 = pr[j, 0]
            py1, px1 = pr[j, 1]
            cost[i, j] = np.hypot(gy0 - py0, gx0 - px0) + np.hypot(gy1 - py1, gx1 - px1)

    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(cost)

    m = 0
    for i, j in zip(r, c):
        if float(cost[i, j]) <= max_cost:
            m += 1
    return m

def match_count_lines_nearest(gt_lines_yx: np.ndarray, pr_lines_yx: np.ndarray, max_cost: float = 15.0) -> int:
    """
    Nearest matching with endpoint swap allowed + Hungarian assignment + threshold.
    Lines are (n,2,2) with points in (y,x).
    Cost = min( L2(s-s)+L2(e-e), L2(s-e)+L2(e-s) ).
    """
    if gt_lines_yx.size == 0 or pr_lines_yx.size == 0:
        return 0

    gt = gt_lines_yx.astype(np.float32)
    pr = pr_lines_yx.astype(np.float32)

    k, l = gt.shape[0], pr.shape[0]
    cost = np.zeros((k, l), dtype=np.float32)

    for i in range(k):
        g0 = gt[i, 0]; g1 = gt[i, 1]
        for j in range(l):
            p0 = pr[j, 0]; p1 = pr[j, 1]
            c_noswap = np.hypot(g0[0]-p0[0], g0[1]-p0[1]) + np.hypot(g1[0]-p1[0], g1[1]-p1[1])
            c_swap   = np.hypot(g0[0]-p1[0], g0[1]-p1[1]) + np.hypot(g1[0]-p0[0], g1[1]-p0[1])
            cost[i, j] = min(c_noswap, c_swap)

    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(cost)

    m = 0
    for i, j in zip(r, c):
        if float(cost[i, j]) <= max_cost:
            m += 1
    return m


def gt_lines_from_Lpos(meta_i, scale: float = 1.75) -> np.ndarray:
    """
    Returns GT lines as array (k,2,2) in same coord order as junc (likely (y,x)),
    scaled to pixel coords.
    """
    junc = meta_i["junc"].detach().cpu().numpy() * scale   # (N,2)
    Lpos = meta_i["Lpos"].detach().cpu().numpy()           # (N,N) 0/1

    ys, xs = np.where(np.triu(Lpos, k=1) > 0)  # upper triangle to avoid duplicates
    if len(ys) == 0:
        return np.zeros((0, 2, 2), dtype=np.float32)

    p1 = junc[ys]  # (k,2)
    p2 = junc[xs]  # (k,2)
    gt = np.stack([p1, p2], axis=1).astype(np.float32)     # (k,2,2)
    return gt

def count_gt_covered_by_pred(
    gt_lines_yx: np.ndarray,   # (k,2,2) points in (y,x)
    pr_lines_yx: np.ndarray,   # (l,2,2)
    max_cost: float = 15.0,    # pixels
) -> int:
    """
    Non-1-to-1 matching:
    m = number of GT lines that have at least one predicted line within max_cost.

    Cost between lines = min( noswap, swap )
      noswap = L2(g0,p0) + L2(g1,p1)
      swap   = L2(g0,p1) + L2(g1,p0)
    """
    if gt_lines_yx.size == 0:
        return 0
    if pr_lines_yx.size == 0:
        return 0

    gt = gt_lines_yx.astype(np.float32)
    pr = pr_lines_yx.astype(np.float32)

    m = 0
    for i in range(gt.shape[0]):
        g0 = gt[i, 0]; g1 = gt[i, 1]

        best = float("inf")
        for j in range(pr.shape[0]):
            p0 = pr[j, 0]; p1 = pr[j, 1]
            c_noswap = np.hypot(g0[0]-p0[0], g0[1]-p0[1]) + np.hypot(g1[0]-p1[0], g1[1]-p1[1])
            c_swap   = np.hypot(g0[0]-p1[0], g0[1]-p1[1]) + np.hypot(g1[0]-p0[0], g1[1]-p0[1])
            c = c_noswap if c_noswap < c_swap else c_swap
            if c < best:
                best = c

        if best <= max_cost:
            m += 1

    return m
