import cv2
import os
import re
import json
import numpy as np
import random
from pathlib import Path
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

from pc.plot_gen.line_data import LineCoordinateExtractor
from pc.plot_gen.plot_utils import safe_join


class CategorySeparator:
    def __init__(self, input_dir, line_coords_json=None):
        self.input_dir = input_dir
        self.line_coords_json = line_coords_json

        with open(self.line_coords_json, 'r') as f:
            self.line_coords_data = json.load(f)

        random.seed(0)
        np.random.seed(0)

    def group_images(self):
        image_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        groups = {}
        for f in image_files:
            base = re.sub(r'_crop_\d+', '', f)
            groups.setdefault(base, []).append(f)
        for k in groups:
            groups[k].sort()
        return groups

    def separate_by_hist_peaks(self, output_dir, output_json="all_histdata.json", color_json="color.json"):
        os.makedirs(output_dir, exist_ok=True)
        output_json = safe_join(output_dir, output_json)
        color_json = safe_join(output_dir, color_json)
        output_data, color_data = [], []
        image_groups = self.group_images()

        for base, crops in image_groups.items():
            first = os.path.join(self.input_dir, crops[0])
            image = cv2.imread(first)
            if image is None:
                continue

            hue = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 0]
            hist, _ = np.histogram(hue, bins=180, range=(0, 180))
            peaks, _ = find_peaks(hist, height=0.05 * np.max(hist), distance=10)
            color_ranges = [(max(0, p - 10), min(180, p + 10)) for p in peaks]

            for crop in crops:
                crop_img_path = os.path.join(self.input_dir, crop)
                image = cv2.imread(crop_img_path)
                if image is None:
                    continue

                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                masks = []
                for peak, (low, high) in zip(peaks, color_ranges):
                    lower_bound = np.array([low, 50, 50], dtype=np.uint8)
                    upper_bound = np.array([high, 255, 255], dtype=np.uint8)
                    mask = cv2.inRange(hsv, lower_bound, upper_bound)
                    masks.append((peak, mask))

                self._process_masks(crop, image, masks, output_data, color_data, output_dir)

        self._save(output_data, color_data, output_json, color_json)

    def separate_by_dbscan(self, output_dir, output_json="all_clusdata.json", color_json="color.json", eps=5, min_samples=200):
        os.makedirs(output_dir, exist_ok=True)
        output_json = safe_join(output_dir, output_json)
        color_json = safe_join(output_dir, color_json)
        output_data, color_data = [], []
        image_groups = self.group_images()

        for base, crops in image_groups.items():
            best_crop, best_clusters, best_factor = None, None, 0
            max_count = 0

            for crop in crops:
                img = cv2.imread(os.path.join(self.input_dir, crop))
                if img is None:
                    continue

                width = img.shape[1]
                resize_factor = 0.3 if width >= 200 else 0.325
                hsv_small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (0, 0), fx=resize_factor, fy=resize_factor)
                hue = hsv_small[:, :, 0].flatten().reshape(-1, 1)
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(hue)
                n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)

                if n_clusters > max_count:
                    max_count = n_clusters
                    best_crop = crop
                    best_clusters = clustering
                    best_factor = resize_factor

            if not best_crop or best_clusters is None:
                continue

            best_image = cv2.imread(os.path.join(self.input_dir, best_crop))
            small = cv2.resize(cv2.cvtColor(best_image, cv2.COLOR_BGR2HSV), (0, 0), fx=best_factor, fy=best_factor)
            hue = small[:, :, 0].flatten().reshape(-1, 1)
            cluster_ids = set(best_clusters.labels_)
            cluster_ids.discard(-1)

            cluster_ranges = {}
            for cid in cluster_ids:
                hues = hue[best_clusters.labels_ == cid]
                if len(hues) == 0:
                    continue
                cluster_ranges[cid] = (int(np.min(hues)), int(np.max(hues)))

            for crop in crops:
                img = cv2.imread(os.path.join(self.input_dir, crop))
                if img is None:
                    continue
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                masks = []
                for cid, (low, high) in cluster_ranges.items():
                    mask = cv2.inRange(hsv, (max(0, low - 5), 50, 50), (min(180, high + 5), 255, 255))
                    hue_avg = (low + high) / 2
                    if np.any(mask):
                        masks.append((cid, mask, hue_avg))

                self._process_masks(crop, img, masks, output_data, color_data, output_dir)

        self._save(output_data, color_data, output_json, color_json)

    def _process_masks(self, crop, image, color_masks, output_data, color_data, output_dir):
        filename = Path(crop).name
        lines = next((x["lines"] for x in self.line_coords_data if x["filename"] == filename), None)
        if not lines:
            return

        color_coords = {k: [] for k, *_ in color_masks}
        for line in lines:
            x1, y1, x2, y2 = map(int, line)
            mid_x, mid_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            for peak, mask, *rest in color_masks:
                if mask[mid_y, mid_x] > 0:
                    color_coords[peak].append(line)
                    break

        for idx, (peak, mask, *rest) in enumerate(color_masks, start=1):
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
                "lines": color_coords[peak]
            })
            color_data.append({
                "filename": out_name,
                "color_hsv": {"h": round(hue_val / 180, 2), "s": 1, "v": 1}
            })

    def _save(self, output_data, color_data, output_json, color_json):
        output_data.sort(key=lambda x: x["filename"])
        color_data.sort(key=lambda x: x["filename"])

        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        os.makedirs(os.path.dirname(color_json), exist_ok=True)

        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=4)

        with open(color_json, 'w') as f:
            json.dump(color_data, f, indent=4)

        print(f" Saved masks → {output_json}")
        print(f"Saved colors → {color_json}")

