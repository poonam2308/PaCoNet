import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import os
import cv2
import json
import re
from pathlib import Path

def extract_crop_cat_index(filename):
    match = re.search(r'_crop_(\d+)_cat_(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return float('inf'), float('inf')  # Sort unknowns last


def compute_hue_histogram(hue_values, bins=180):
    return np.histogram(hue_values, bins=bins, range=(0, 180))[0]

def detect_peaks(hist, min_height_ratio=0.005, distance=5):
    peaks, _ = find_peaks(hist, height=min_height_ratio * np.max(hist), distance=distance)
    return peaks

def get_color_ranges(peaks, tolerance=20):
    return [(max(0, p - tolerance), min(179, p + tolerance)) for p in peaks]

def create_color_mask(hsv_img, low, high, s_thresh=20, v_thresh=1):
    if low > high:
        mask1 = cv2.inRange(hsv_img, (0, s_thresh, v_thresh), (high, 255, 255))
        mask2 = cv2.inRange(hsv_img, (low, s_thresh, v_thresh), (179, 255, 255))
        return cv2.bitwise_or(mask1, mask2)
    else:
        return cv2.inRange(hsv_img, (int(low), int(s_thresh), int(v_thresh)), (int(high), 255, 255))

def apply_mask_with_white_background(image, mask):
    white_bg = np.full_like(image, 255)
    masked = cv2.bitwise_and(image, image, mask=mask)
    inverse = cv2.bitwise_not(mask)
    return cv2.bitwise_or(masked, cv2.bitwise_and(white_bg, white_bg, mask=inverse))

def group_by_base_name(file_list):
    groups = {}
    for fname in file_list:
        base = re.sub(r'_crop_\d+', '', fname.split('.')[0])
        groups.setdefault(base, []).append(fname)
    return groups

# def ensure_red_peak(peaks):
#     peaks = list(peaks)
#     if not any(p <= 10 or p >= 170 for p in peaks):
#         peaks.append(0)
#     return sorted(list(set(peaks)))


def ensure_red_peak(peaks, red_tolerance=10):
    """
    Ensures that a red peak is considered. If no existing peak is within the red hue range
    (0-red_tolerance or 180-red_tolerance), it adds 0 as a representative red peak.
    """
    peaks_list = list(peaks)

    red_peak_exists = False
    for p in peaks_list:
        if (p >= 0 and p <= red_tolerance) or (p >= (180 - red_tolerance) and p <= 179):
            red_peak_exists = True
            break

    if not red_peak_exists:
        peaks_list.append(0)  # Add a representative red peak if none exists

    return sorted(list(set(peaks_list)))


def merge_close_peaks(peaks, merge_tolerance=15):
    """
    Merges peaks that are closer than a specified tolerance.
    Prioritizes merging peaks that are effectively the same color.
    Handles wrap-around for hue (0 and 179 being close).
    """
    if not peaks:
        return []

    peaks = sorted(list(set(peaks)))  # Ensure unique and sorted
    if not peaks:  # Check again after set conversion if all were duplicates
        return []

    merged_peaks = [peaks[0]]

    for i in range(1, len(peaks)):
        current_peak = peaks[i]
        last_merged_peak = merged_peaks[-1]

        # Check for standard proximity
        if abs(current_peak - last_merged_peak) <= merge_tolerance:
            continue  # Skip adding if too close to the last merged peak
        # Check for wrap-around proximity (e.g., 175 and 5, considering 0-179 range)
        elif (current_peak < merge_tolerance and last_merged_peak > (179 - merge_tolerance)):  # Example: 5 and 175
            continue  # Skip if current is near 0 and last is near 179
        elif (last_merged_peak < merge_tolerance and current_peak > (179 - merge_tolerance)):  # Example: 175 and 5
            continue  # Skip if last is near 0 and current is near 179
        else:
            merged_peaks.append(current_peak)

    # Final check for wrap-around red specifically if 0 was added and 179 was a separate peak
    if len(merged_peaks) > 1:
        if merged_peaks[0] == 0 and any(p >= (180 - merge_tolerance) for p in merged_peaks[1:]):
            # If 0 is present and another peak is near 179, remove the near-179 one
            # to prevent duplicate red. This assumes 0 is the canonical red for merging.
            merged_peaks = [p for p in merged_peaks if not (p >= (180 - merge_tolerance) and p != 0)]
        elif merged_peaks[-1] == 179 and any(p <= merge_tolerance for p in merged_peaks[:-1]):
            # If 179 is present and another peak is near 0, remove the near-0 one
            # This logic might need to be fine-tuned based on desired red representation
            merged_peaks = [p for p in merged_peaks if not (p <= merge_tolerance and p != 179)]

    return sorted(list(set(merged_peaks)))  # Use set to remove potential duplicates after specific red handling


def save_histogram(hist, peaks, name, output_dir):
    try:
        plt.figure(figsize=(8, 3))
        plt.plot(hist)
        plt.scatter(peaks, hist[peaks], color='red')
        plt.title("Hue Histogram with Detected Peaks")
        plt.xlabel("Hue")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_hist.png"))
        plt.close()
    except Exception as e:
        print(f"Could not save hue histogram: {e}")

def process_images_separation(input_dir, output_dir, output_json, method='peaks', top_k=3):
    os.makedirs(output_dir, exist_ok=True)
    output_data = []

    if os.path.exists(output_json):
        with open(output_json, 'r') as f:
            output_data = json.load(f)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    groups = group_by_base_name(image_files)

    for base_name, crops in groups.items():
        crops.sort()
        hue_values = []

        for crop in crops:
            image = cv2.imread(os.path.join(input_dir, crop))
            if image is None:
                continue
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            mask = (s > 20) & (v > 1)
            hue_values.extend(h[mask])

        hue_values = np.array(hue_values)
        if hue_values.size == 0:
            print(f"Skipped group {base_name} (no hue data across crops).")
            continue

        hist = compute_hue_histogram(hue_values)

        if method == 'topk':
            top_indices = np.argsort(hist)[-top_k:]
            peaks = sorted(list(top_indices))
            peaks = ensure_red_peak(peaks)
            peaks = merge_close_peaks(peaks)  # Merge close peaks
        elif method == 'peaks':
            peaks = detect_peaks(hist)
            peaks = ensure_red_peak(peaks)
            peaks = merge_close_peaks(peaks)  # Merge close peaks
        else:
            raise ValueError(f"Unsupported method: {method}")

        # Save histogram for debugging/visualization (optional)
        # save_histogram(hist, peaks, base_name, output_dir)

        color_ranges = get_color_ranges(peaks)

        for crop in crops:
            crop_img = cv2.imread(os.path.join(input_dir, crop))
            if crop_img is None:
                continue
            hsv_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
            seen_masks=[]
            for idx, (peak, (low, high)) in enumerate(zip(peaks, color_ranges), start=1):
                mask = create_color_mask(hsv_crop, low, high)
                nonzero_ratio = np.count_nonzero(mask) / mask.size
                if nonzero_ratio < 0.01:
                    continue
                    # Check for similarity with previous masks
                is_duplicate = any(np.allclose(mask, prev_mask) for prev_mask in seen_masks)
                if is_duplicate:
                    continue
                seen_masks.append(mask)

                result = apply_mask_with_white_background(crop_img, mask)
                fname = f"{Path(crop).stem}_cat_{idx}.png"
                out_path = os.path.join(output_dir, fname)
                cv2.imwrite(out_path, result)

                output_data.append({
                    "filename": fname,
                    "color_hsv": {"h": round(peak / 180, 2), "s": 1, "v": 1}
                })

    output_data.sort(key=lambda x: x["filename"])
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Saved {len(output_data)} entries to {output_json}")

    output_data.sort(key=lambda x: extract_crop_cat_index(x["filename"]))

    # Save just hues in order of crop/cat
    hue_list = [
        {
            "filename": item["filename"],
            "hue": item["color_hsv"]["h"]
        }
        for item in output_data
    ]

    with open(os.path.join(output_dir, "hue_summary.json"), "w") as f:
        json.dump(hue_list, f, indent=2)

