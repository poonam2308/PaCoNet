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

def get_color_ranges(peaks, tolerance=15):
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

def ensure_red_peak(peaks):
    peaks = list(peaks)
    if not any(p <= 10 or p >= 170 for p in peaks):
        peaks.append(0)
    return sorted(list(set(peaks)))

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
        elif method == 'peaks':
            peaks = detect_peaks(hist)
            peaks = ensure_red_peak(peaks)
        else:
            raise ValueError(f"Unsupported method: {method}")

        color_ranges = get_color_ranges(peaks)

        for crop in crops:
            crop_img = cv2.imread(os.path.join(input_dir, crop))
            if crop_img is None:
                continue
            hsv_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

            for idx, (peak, (low, high)) in enumerate(zip(peaks, color_ranges), start=1):
                mask = create_color_mask(hsv_crop, low, high)
                nonzero_ratio = np.count_nonzero(mask) / mask.size
                if nonzero_ratio < 0.01:
                    continue

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

