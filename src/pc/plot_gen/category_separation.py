
import cv2
import os
import re
import json
import numpy as np
import random
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

class CategorySeparator:
    def __init__(self):
        random.seed(0)
        np.random.seed(0)

    def remove_black_background(img_np, method=1, v_thresh=40):
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2]
        mask = v < v_thresh
        if method == 1:
            img_np[mask] = [255, 255, 255]
        elif method == 2:
            mask_uint8 = mask.astype(np.uint8) * 255
            img_np = cv2.inpaint(img_np, mask_uint8, 3, cv2.INPAINT_TELEA)
        return img_np

    def _load_lines(self, json_path, crop_filename):
        if not os.path.exists(json_path):
            print(f"JSON not found: {json_path}")
            return []

        with open(json_path, "r") as f:
            data = json.load(f)

        m = re.search(r"_crop_(\d+)", Path(crop_filename).stem)
        if not m:
            return []

        crop_key = f"crop_{m.group(1)}"
        lines_dict = data.get("lines", {})
        return lines_dict.get(crop_key, [])

    def _process_masks(self, crop, image, color_masks, lines, output_dir):
        output_data, color_data = [], []
        cat_coords = {k: [] for k, *_ in color_masks}

        for line in lines:
            x1, y1, x2, y2 = map(int, line)
            mid_x, mid_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            for peak, mask, *rest in color_masks:
                if mask[mid_y, mid_x] > 0:
                    cat_coords[peak].append(line)
                    break
        for  idx, (peak, mask, *rest) in enumerate(color_masks, start=1):
            hue_val = rest[0] if rest else peak
            white = np.full_like(image, 255)
            masked = cv2.bitwise_and(image, image, mask=mask)
            inv = cv2.bitwise_not(mask)
            result = cv2.bitwise_or(masked, cv2.bitwise_and(white, white, mask=inv))

            out_name = f"{Path(crop).stem}_cat_{idx}.png"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, result)

            output_data.append({
                "filename": out_name,
                "lines": cat_coords[peak]
            })
            color_data.append({
                "filename": out_name,
                "color_hsv": {"h": round(hue_val / 180, 2), "s": 1, "v": 1}
            })

        return output_data, color_data

    def process_single_image(self, image_path, json_path, output_dir, method="hist",save_per_file=False, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read {image_path}")
            return [], []

        lines = self._load_lines(json_path, Path(image_path).name)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masks = []

        if method == "hist":
            hue = hsv[:, :, 0]
            hist, _ = np.histogram(hue, bins=180, range=(0, 180))
            peaks, _ = find_peaks(hist, height=0.05 * np.max(hist), distance=10)
            for peak in peaks:
                lower_bound = np.array([max(0, peak - 10), 50, 50], dtype=np.uint8)
                upper_bound = np.array([min(180, peak + 10), 255, 255], dtype=np.uint8)
                mask = cv2.inRange(hsv, lower_bound, upper_bound)
                masks.append((peak, mask))

        elif method == "dbscan":
            eps = kwargs.get("eps", 5)
            min_samples = kwargs.get("min_samples", 200)
            resize_factor = 0.3 if image.shape[1] >= 200 else 0.325
            hsv_small = cv2.resize(hsv, (0, 0), fx=resize_factor, fy=resize_factor)
            hue = hsv_small[:, :, 0].flatten().reshape(-1, 1)
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(hue)
            cluster_ids = set(clustering.labels_);
            cluster_ids.discard(-1)
            for cid in cluster_ids:
                hues = hue[clustering.labels_ == cid]
                if len(hues) > 0:
                    low, high = int(np.min(hues)), int(np.max(hues))
                    mask = cv2.inRange(hsv, (max(0, low - 5), 50, 50),
                                       (min(180, high + 5), 255, 255))
                    hue_avg = (low + high) / 2
                    if np.any(mask):
                        masks.append((cid, mask, hue_avg))

        cat_lines_data, color_data = self._process_masks(image_path, image, masks, lines, output_dir)

        # 🔹 Save JSONs
        if save_per_file:  # 🔹 only save if flag is True
            base_name = Path(image_path).stem
            with open(os.path.join(output_dir, f"{base_name}_output.json"), "w") as f:
                json.dump(cat_lines_data, f, indent=4)
            with open(os.path.join(output_dir, f"{base_name}_colors.json"), "w") as f:
                json.dump(color_data, f, indent=4)

        return cat_lines_data, color_data

    def process_single_image_enhanced(self, image_path, json_path, output_dir,
                                      bg_method=None, sat_thresh=50, save_per_file=False,
                                      show_plot=False):
        os.makedirs(output_dir, exist_ok=True)
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)

        if bg_method in [1, 2]:
            img_np = self.remove_black_background(img_np, method=bg_method)

        img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        hue = img_hsv[:, :, 0].flatten()
        sat = img_hsv[:, :, 1].flatten()
        mask = sat > sat_thresh
        hue_filtered = hue[mask]

        hist, bins = np.histogram(hue_filtered, bins=180, range=(0, 180))
        peaks, _ = find_peaks(hist, height=np.max(hist) * 0.07, distance=5)

        masks = []
        for p in peaks:
            low, high = max(0, p - 10), min(180, p + 10)
            lower = np.array([low, 50, 50], dtype=np.uint8)
            upper = np.array([high, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(img_hsv, lower, upper)
            masks.append((p, mask))

        lines = self._load_lines(json_path, Path(image_path).name)
        cat_lines_data, color_data = self._process_masks(
            image_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
            masks, lines, output_dir
        )

        if show_plot:
            plt.figure(figsize=(10, 4))
            plt.plot(hist, label="Hue Histogram")
            plt.plot(peaks, hist[peaks], "rx", label="Detected Peaks")
            plt.title(f"Hue Histogram with Peaks – {Path(image_path).name}")
            plt.xlabel("Hue Value (0-179)")
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()

        # 🔹 Save JSONs
        if save_per_file:  # 🔹 only save if flag is True
            base_name = Path(image_path).stem
            with open(os.path.join(output_dir, f"{base_name}_output.json"), "w") as f:
                json.dump(cat_lines_data, f, indent=4)
            with open(os.path.join(output_dir, f"{base_name}_colors.json"), "w") as f:
                json.dump(color_data, f, indent=4)

        return cat_lines_data, color_data

    def process_batch(self, input_dir, json_dir, output_dir,
                      method="hist_enhanced", **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        all_output, all_colors = [], []

        for f in os.listdir(input_dir):
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(input_dir, f)
            base = re.sub(r'_crop_\d+', '', Path(f).stem)  # strip _crop_x
            json_path = os.path.join(json_dir, base + ".json") # meta data json dir

            if method == "hist_enhanced":
                result, colors = self.process_single_image_enhanced(
                    image_path, json_path, output_dir, **kwargs)
            else:
                result, colors = self.process_single_image(
                    image_path, json_path, output_dir, method, **kwargs)

            all_output.extend(result)
            all_colors.extend(colors)

        with open(os.path.join(output_dir, "all_output.json"), "w") as f:
            json.dump(all_output, f, indent=4)
        with open(os.path.join(output_dir, "all_colors.json"), "w") as f:
            json.dump(all_colors, f, indent=4)

        print(f"Saved batch results in {output_dir}")
        # return all_output, all_colors


