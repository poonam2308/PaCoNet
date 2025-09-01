import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch


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



