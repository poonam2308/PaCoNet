import cv2
import os
import re
import json
import numpy as np
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

class CategorySeparator:
    """
    Minimal, opinionated version:
      - Only process_single_image_enhanced is exposed.
      - Background is **always white**.
      - Works with or without a JSON. If JSON has `category_colors`, we map peaks to closest category.
      - Peak detection: histogram on (optionally saturation-filtered) hue.
      - Supports `top_k` to keep only the K strongest hue peaks (by histogram height).
    """

    # ---------- Utilities ----------
    @staticmethod
    def _json_exists(json_path):
        return bool(json_path) and os.path.exists(json_path)

    @staticmethod
    def _closest_category(hsv_val, category_colors):
        # hsv_val: (h, s, v) with h in [0,1]
        best_cat, best_dist = None, float("inf")
        for cat, ref in category_colors.items():
            ref_h, ref_s, ref_v = ref["h"], ref["s"], ref["v"]
            dist = (hsv_val[0] - ref_h) ** 2 + (hsv_val[1] - ref_s) ** 2 + (hsv_val[2] - ref_v) ** 2
            if dist < best_dist:
                best_cat, best_dist = cat, dist
        return best_cat

    def _load_lines(self, json_path, crop_filename):
        """Load line coordinates for a specific crop file. Returns [] if json missing/unmatched."""
        if not self._json_exists(json_path):
            return []
        with open(json_path, "r") as f:
            data = json.load(f)

        m = re.search(r"_crop_(\d+)", Path(crop_filename).stem)
        if not m:
            return []

        crop_key = f"crop_{m.group(1)}"
        lines_dict = data.get("lines", {})

        crop_lines = lines_dict.get(crop_key, [])
        if isinstance(crop_lines, dict):
            # New format → merge all category lists
            merged = []
            for _, coords in crop_lines.items():
                merged.extend(coords)
            return merged
        else:
            return crop_lines

    # ---------- Peak detection (histogram) ----------
    @staticmethod
    def _build_peak_masks_from_hist(hsv_img, peaks, tol=10):
        masks = []
        for p in peaks:
            low, high = max(0, int(p) - tol), min(180, int(p) + tol)
            lower = np.array([low, 50, 50], dtype=np.uint8)
            upper = np.array([high, 255, 255], dtype=np.uint8)
            m = cv2.inRange(hsv_img, lower, upper)
            masks.append((int(p), m))
        return masks

    # ---------- Mask processing ----------
    def _process_masks_simple_white_bg(self, crop, image_bgr, color_masks, lines, output_dir):
        """
        No JSON categories: save *_cat_#.png with WHITE background.
        """
        output_data, color_data = [], []
        cat_coords = {k: [] for k, *_ in color_masks}

        for line in lines or []:
            x1, y1, x2, y2 = map(int, line)
            mid_x, mid_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            for peak, mask, *rest in color_masks:
                if 0 <= mid_y < mask.shape[0] and 0 <= mid_x < mask.shape[1] and mask[mid_y, mid_x] > 0:
                    cat_coords[peak].append(line)
                    break

        for idx, (peak, mask, *rest) in enumerate(color_masks, start=1):
            white = np.full_like(image_bgr, 255)
            fg = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
            inv = cv2.bitwise_not(mask)
            result = cv2.bitwise_or(fg, cv2.bitwise_and(white, white, mask=inv))

            out_name = f"{Path(crop).stem}_cat_{idx}.png"
            cv2.imwrite(os.path.join(output_dir, out_name), result)

            output_data.append({"filename": out_name, "lines": cat_coords.get(peak, [])})
            color_data.append({
                "filename": out_name,
                "color_hsv": {"h": round(peak / 180, 2), "s": 1, "v": 1}
            })
        return output_data, color_data

    def _process_masks_categories_white_bg(self, crop, image_bgr, color_masks, lines, output_dir, category_colors):
        """
        JSON categories present: group by closest configured category, WHITE background only.
        """
        output_data, color_data = [], []
        cat_coords = {cat: [] for cat in (category_colors or {})}
        cat_masks = {cat: np.zeros(image_bgr.shape[:2], dtype=np.uint8) for cat in (category_colors or {})}

        # Assign lines and build cat masks
        for line in lines or []:
            x1, y1, x2, y2 = map(int, line)
            mid_x, mid_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            for peak, mask, *rest in color_masks:
                if 0 <= mid_y < mask.shape[0] and 0 <= mid_x < mask.shape[1] and mask[mid_y, mid_x] > 0:
                    hsv_val = (peak / 180, 1, 1)
                    cat = self._closest_category(hsv_val, category_colors)
                    cat_coords.setdefault(cat, []).append(line)
                    cat_masks[cat] = cv2.bitwise_or(cat_masks[cat], mask)
                    break

        # Save per-category results (white bg)
        for cat, coords in cat_coords.items():
            mask = cat_masks.get(cat, None)
            if mask is None or np.count_nonzero(mask) == 0:
                continue

            white = np.full_like(image_bgr, 255)
            fg = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
            inv = cv2.bitwise_not(mask)
            result = cv2.bitwise_or(fg, cv2.bitwise_and(white, white, mask=inv))

            out_name = f"{Path(crop).stem}_{cat}.png"
            cv2.imwrite(os.path.join(output_dir, out_name), result)

            output_data.append({"filename": out_name, "lines": coords})
            color_data.append({"filename": out_name, "color_hsv": category_colors[cat]})
        return output_data, color_data

    # ---------- Public API (only this) ----------
    def process_single_image_enhanced(
        self,
        image_path,
        json_path=None,
        output_dir=".",
        sat_thresh=50,
        save_per_file=False,
        show_plot=False,
        top_k=None,                 # <-- NEW: limit number of peaks to keep
        peak_height_frac=0.05,      # keep defaults aligned with previous enhanced flow
        peak_distance=5,
        tolerance=10
    ):
        """
        Enhanced variant with saturation filtering. Always uses WHITE background.
        Works with or without json. If `category_colors` exists in JSON, results are grouped by category.

        Peak detection:
          - Build hue histogram (0..179) from saturation-filtered pixels
          - Detect peaks with height = peak_height_frac * max(hist) and distance = peak_distance
          - Optionally keep only top_k peaks by histogram height
          - Build masks with ±tolerance around each peak
        """
        os.makedirs(output_dir, exist_ok=True)

        # Load and convert
        pil_img = Image.open(image_path).convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # Optional JSON inputs
        lines = self._load_lines(json_path, Path(image_path).name) if self._json_exists(json_path) else []
        category_colors = None
        if self._json_exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            category_colors = data.get("category_colors", None)

        # Saturation filter then histogram on hue
        hue = hsv[:, :, 0].flatten()
        sat = hsv[:, :, 1].flatten()
        valid = sat > sat_thresh
        hue_filtered = hue[valid]

        hist, _ = np.histogram(hue_filtered, bins=180, range=(0, 180))
        peaks = np.array([], dtype=int)
        if np.max(hist) > 0:
            peaks, _ = find_peaks(hist, height=np.max(hist) * float(peak_height_frac), distance=int(peak_distance))

            # If top_k specified, keep the strongest K by histogram height
            if top_k is not None:
                try:
                    k = int(top_k)
                except Exception:
                    k = None
                if k is not None and k > 0 and len(peaks) > k:
                    # sort peaks by their histogram heights descending
                    order = np.argsort(hist[peaks])[::-1]
                    peaks = peaks[order[:k]]

        masks = self._build_peak_masks_from_hist(hsv, peaks, tol=int(tolerance))

        # Process with white background only
        if category_colors:
            out_data, color_data = self._process_masks_categories_white_bg(
                image_path, img_bgr, masks, lines, output_dir, category_colors
            )
        else:
            out_data, color_data = self._process_masks_simple_white_bg(
                image_path, img_bgr, masks, lines, output_dir
            )

        if show_plot:
            plt.figure(figsize=(10, 4))
            plt.plot(hist, label="Hue Histogram")
            if len(peaks) > 0:
                plt.plot(peaks, hist[peaks], "rx", label="Detected Peaks")
            plt.title(f"Hue Histogram with Peaks – {Path(image_path).name}")
            plt.xlabel("Hue Value (0–179)")
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()

        if save_per_file:
            base = Path(image_path).stem
            with open(os.path.join(output_dir, f"{base}_output.json"), "w") as f:
                json.dump(out_data, f, indent=4)
            with open(os.path.join(output_dir, f"{base}_colors.json"), "w") as f:
                json.dump(color_data, f, indent=4)

        return out_data, color_data

    def process_batch(self, input_dir, json_dir=None, output_dir=".", **kwargs):
        """
        Batch version using ONLY the enhanced method and WHITE background.
        If json_dir provided, we try to find a matching base JSON per file. Otherwise we run without JSON.
        Pass-through for parameters like `top_k`, `sat_thresh`, etc.
        """
        os.makedirs(output_dir, exist_ok=True)
        all_output, all_colors = [], []

        for f in os.listdir(input_dir):
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(input_dir, f)
            base = re.sub(r'_crop_\d+', '', Path(f).stem)  # strip _crop_x
            json_path = None
            if json_dir:
                cand = os.path.join(json_dir, base + ".json")
                if os.path.exists(cand):
                    json_path = cand

            result, colors = self.process_single_image_enhanced(
                image_path=image_path,
                json_path=json_path,
                output_dir=output_dir,
                **kwargs
            )
            all_output.extend(result)
            all_colors.extend(colors)

        with open(os.path.join(output_dir, "all_data.json"), "w") as f:
            json.dump(all_output, f, indent=4)
        with open(os.path.join(output_dir, "all_colors.json"), "w") as f:
            json.dump(all_colors, f, indent=4)

        print(f"Saved batch results in {output_dir}")
