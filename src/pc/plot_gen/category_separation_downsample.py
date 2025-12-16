import os
import re
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.signal import find_peaks


class DownsampledHistogramBatchSeparator:
    """
    Histogram peak-based separator with:
      - Downsample ONLY for peak detection (histogram + peaks)
      - Full-res masking + full-res outputs (white background)
      - Batch grouping by base (strip crop tokens), choose "best" crop by #peaks
      - cluster_vs_gt.json + cluster_vs_gt_summary.json

    Compatible with your JSON format:
      - data["lines"]["crop_k"] can be list OR dict(cat->coords)
      - data["category_colors"] optional
    """

    # ---------- Utilities ----------
    @staticmethod
    def _json_exists(p):
        return bool(p) and os.path.exists(p)

    @staticmethod
    def _base_stem_like_clustering(stem: str) -> str:
        """Remove tokens 'crop' + number (clustering-style)."""
        toks = [t for t in stem.split("_") if t]
        out = []
        i = 0
        while i < len(toks):
            if toks[i].lower() == "crop" and i + 1 < len(toks) and toks[i + 1].isdigit():
                i += 2
            else:
                out.append(toks[i])
                i += 1
        return "_".join(out)

    @staticmethod
    def _closest_category(hsv_val_01, category_colors):
        # hsv_val_01: (h,s,v) each in [0..1]
        best_cat, best_dist = None, float("inf")
        for cat, ref in category_colors.items():
            rh, rs, rv = ref["h"], ref["s"], ref["v"]
            d = (hsv_val_01[0] - rh) ** 2 + (hsv_val_01[1] - rs) ** 2 + (hsv_val_01[2] - rv) ** 2
            if d < best_dist:
                best_cat, best_dist = cat, d
        return best_cat

    def _load_lines_structured(self, json_path, crop_filename):
        """
        Returns (per_category_dict_or_None, merged_list).
        If lines[crop_k] is a dict: returns (dict, merged list in dict order)
        If it's a list: returns (None, list)
        """
        if not self._json_exists(json_path):
            return None, []

        with open(json_path, "r") as f:
            data = json.load(f)

        m = re.search(r"_crop_(\d+)", Path(crop_filename).stem)
        if not m:
            return None, []

        crop_key = f"crop_{m.group(1)}"
        crop_lines = (data.get("lines", {}) or {}).get(crop_key, [])

        if isinstance(crop_lines, dict):
            merged = []
            for _, coords in (crop_lines or {}).items():
                if coords:
                    merged.extend(coords)
            return crop_lines, merged
        return None, (crop_lines or [])

    def _count_gt_categories_for_base(self, json_path, crops):
        """
        Count GT categories present (non-empty) across crop_* for this base.
        Only counts when lines[crop_k] is a dict(cat->coords).
        """
        if not self._json_exists(json_path):
            return 0
        with open(json_path, "r") as f:
            data = json.load(f)

        lines_root = (data.get("lines") or {})
        present = set()

        for crop_name in crops:
            m = re.search(r"_crop_(\d+)", Path(crop_name).stem)
            if not m:
                continue
            crop_key = f"crop_{m.group(1)}"
            v = lines_root.get(crop_key, [])
            if isinstance(v, dict):
                for cat, coords in v.items():
                    if coords and len(coords) > 0:
                        present.add(cat)
        return len(present)

    # ---------- Peak detection ----------
    @staticmethod
    def _detect_peaks_from_image(
        img_rgb_full,
        sat_thresh=50,
        peak_height_frac=0.05,
        peak_distance=5,
        top_k=None,
        resize_factor=0.30,
        interp=cv2.INTER_NEAREST,
    ):
        """
        Returns (peaks, hist_max) where peaks are hue-bin indices (0..179).
        Downsamples ONLY for histogram computation if resize_factor < 1.
        """
        hsv_full = cv2.cvtColor(img_rgb_full, cv2.COLOR_RGB2HSV)
        hsv_for_hist = hsv_full
        if resize_factor is not None and float(resize_factor) < 1.0:
            hsv_for_hist = cv2.resize(
                hsv_full, (0, 0),
                fx=float(resize_factor), fy=float(resize_factor),
                interpolation=interp
            )

        hue = hsv_for_hist[:, :, 0].reshape(-1)
        sat = hsv_for_hist[:, :, 1].reshape(-1)
        valid = sat > int(sat_thresh)
        hue_filtered = hue[valid]

        hist, _ = np.histogram(hue_filtered, bins=180, range=(0, 180))
        if np.max(hist) <= 0:
            return np.array([], dtype=int), 0

        peaks, _ = find_peaks(
            hist,
            height=np.max(hist) * float(peak_height_frac),
            distance=int(peak_distance),
        )

        # keep strongest top_k peaks by histogram height
        if top_k is not None:
            k = int(top_k)
            if k > 0 and len(peaks) > k:
                order = np.argsort(hist[peaks])[::-1]
                peaks = peaks[order[:k]]

        return peaks.astype(int), int(np.max(hist))

    @staticmethod
    def _build_peak_masks_fullres(hsv_full, peaks, tolerance=10):
        """
        Build masks on FULL resolution HSV from peak positions.
        Returns list of tuples: (peak, mask)
        """
        masks = []
        for p in peaks:
            low = max(0, int(p) - int(tolerance))
            high = min(179, int(p) + int(tolerance))
            lower = np.array([low, 50, 50], dtype=np.uint8)
            upper = np.array([high, 255, 255], dtype=np.uint8)
            m = cv2.inRange(hsv_full, lower, upper)
            if np.count_nonzero(m) == 0:
                continue
            masks.append((int(p), m))
        # stable order: increasing hue peak
        masks.sort(key=lambda t: t[0])
        return masks

    # ---------- Apply masks to one crop ----------
    def _apply_peak_masks_to_crop(
        self,
        crop_path,
        peak_masks,             # list[(peak, mask)] masks already for THIS crop (full-res)
        json_path,
        output_dir,
        category_colors=None,
        prefer_gt_lines=True,
    ):
        """
        Save white-bg cutouts + assign lines.
        - If category_colors exists: merge peak masks into category masks via closest_category
        - If prefer_gt_lines and per-category GT exists: use GT lines verbatim per category
        - Else: assign lines by midpoint-in-mask
        Returns (output_data, color_data)
        """
        os.makedirs(output_dir, exist_ok=True)

        img_bgr = cv2.imread(crop_path)
        if img_bgr is None:
            raise FileNotFoundError(crop_path)

        # load lines
        gt_cat_dict = None
        merged_lines = []
        if self._json_exists(json_path):
            gt_cat_dict, merged_lines = self._load_lines_structured(json_path, Path(crop_path).name)

        output_data, color_data = [], []

        # ---------- Category mode ----------
        if category_colors:
            cat_coords = {cat: [] for cat in category_colors}
            cat_masks = {cat: np.zeros(img_bgr.shape[:2], dtype=np.uint8) for cat in category_colors}

            if prefer_gt_lines and isinstance(gt_cat_dict, dict):
                for cat in cat_coords:
                    cat_coords[cat] = gt_cat_dict.get(cat, [])

                # build visual masks by mapping each peak to nearest category
                for peak, m in peak_masks:
                    cat = self._closest_category((peak / 180.0, 1, 1), category_colors)
                    cat_masks[cat] = cv2.bitwise_or(cat_masks[cat], m)
            else:
                # assign by midpoint
                for line in (merged_lines or []):
                    x1, y1, x2, y2 = map(int, line)
                    mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    for peak, m in peak_masks:
                        if 0 <= my < m.shape[0] and 0 <= mx < m.shape[1] and m[my, mx] > 0:
                            cat = self._closest_category((peak / 180.0, 1, 1), category_colors)
                            cat_coords.setdefault(cat, []).append(line)
                            cat_masks[cat] = cv2.bitwise_or(cat_masks[cat], m)
                            break

            # save
            for cat, m in cat_masks.items():
                if np.count_nonzero(m) == 0:
                    continue
                white = np.full_like(img_bgr, 255)
                fg = cv2.bitwise_and(img_bgr, img_bgr, mask=m)
                inv = cv2.bitwise_not(m)
                out = cv2.bitwise_or(fg, cv2.bitwise_and(white, white, mask=inv))

                out_name = f"{Path(crop_path).stem}_{cat}.png"
                cv2.imwrite(os.path.join(output_dir, out_name), out)
                output_data.append({"filename": out_name, "lines": cat_coords.get(cat, [])})
                color_data.append({"filename": out_name, "color_hsv": category_colors[cat]})

            return output_data, color_data

        # ---------- No category_colors mode: *_cat_# ----------
        cat_coords = {peak: [] for peak, _ in peak_masks}
        for line in (merged_lines or []):
            x1, y1, x2, y2 = map(int, line)
            mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
            for peak, m in peak_masks:
                if 0 <= my < m.shape[0] and 0 <= mx < m.shape[1] and m[my, mx] > 0:
                    cat_coords[peak].append(line)
                    break

        for idx, (peak, m) in enumerate(peak_masks, start=1):
            white = np.full_like(img_bgr, 255)
            fg = cv2.bitwise_and(img_bgr, img_bgr, mask=m)
            inv = cv2.bitwise_not(m)
            out = cv2.bitwise_or(fg, cv2.bitwise_and(white, white, mask=inv))

            out_name = f"{Path(crop_path).stem}_cat_{idx}.png"
            cv2.imwrite(os.path.join(output_dir, out_name), out)
            output_data.append({"filename": out_name, "lines": cat_coords.get(peak, [])})
            color_data.append({"filename": out_name, "color_hsv": {"h": round(peak / 180.0, 2), "s": 1, "v": 1}})

        return output_data, color_data

    # ---------- Public: single image (self-contained) ----------
    def process_single_image(
        self,
        image_path,
        json_path=None,
        output_dir=".",
        sat_thresh=50,
        peak_height_frac=0.05,
        peak_distance=5,
        top_k=None,
        tolerance=10,
        resize_factor=0.30,
        prefer_gt_lines=True,
    ):
        """
        Discover peaks (downsampled histogram), then apply masks on full-res image.
        """
        os.makedirs(output_dir, exist_ok=True)

        pil_img = Image.open(image_path).convert("RGB")
        img_rgb = np.array(pil_img)
        hsv_full = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # optional categories
        category_colors = None
        if self._json_exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            category_colors = data.get("category_colors", None)

        peaks, _ = self._detect_peaks_from_image(
            img_rgb_full=img_rgb,
            sat_thresh=sat_thresh,
            peak_height_frac=peak_height_frac,
            peak_distance=peak_distance,
            top_k=top_k,
            resize_factor=resize_factor,
        )
        peak_masks = self._build_peak_masks_fullres(hsv_full, peaks, tolerance=tolerance)

        # save using same apply logic by writing a temp file? (avoid that)
        # We'll just reuse _apply_peak_masks_to_crop() which expects cv2.imread.
        # So we save nothing here; for single-image path, just re-read via cv2 in apply.
        return self._apply_peak_masks_to_crop(
            crop_path=image_path,
            peak_masks=peak_masks,
            json_path=json_path,
            output_dir=output_dir,
            category_colors=category_colors,
            prefer_gt_lines=prefer_gt_lines,
        )

    # ---------- Public: batch (group by base; choose best crop) ----------
    def process_batch(
        self,
        input_dir,
        json_dir=None,
        output_dir=".",
        sat_thresh=50,
        peak_height_frac=0.05,
        peak_distance=5,
        top_k=None,
        tolerance=10,
        width_threshold=200,
        resize_factor_large=0.30,
        resize_factor_small=0.325,
        prefer_gt_lines=True,
    ):
        """
        Batch mode (like clustering batch):
          - group by base (strip crop tokens)
          - choose "best crop" = most detected peaks
          - compute peaks on best crop (downsampled hist)
          - for each crop: build full-res masks using those peaks, save cutouts
          - write all_data/all_colors + cluster_vs_gt + summary
        """
        os.makedirs(output_dir, exist_ok=True)

        # inventory + group
        files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        groups = {}
        for f in files:
            base = self._base_stem_like_clustering(Path(f).stem)
            groups.setdefault(base, []).append(f)

        all_output, all_colors = [], []
        comparisons = []

        for base, crops in groups.items():
            crops.sort()

            # base json
            json_path = None
            if json_dir:
                cand = os.path.join(json_dir, base + ".json")
                if os.path.exists(cand):
                    json_path = cand

            category_colors = None
            if self._json_exists(json_path):
                with open(json_path, "r") as f:
                    data = json.load(f)
                category_colors = data.get("category_colors", None)

            # choose best crop = max peaks
            best_crop = None
            best_peaks = np.array([], dtype=int)
            best_count = -1

            for crop in crops:
                p = os.path.join(input_dir, crop)
                img_bgr = cv2.imread(p)
                if img_bgr is None:
                    continue

                rf = resize_factor_large if img_bgr.shape[1] >= width_threshold else resize_factor_small
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                peaks, _ = self._detect_peaks_from_image(
                    img_rgb_full=img_rgb,
                    sat_thresh=sat_thresh,
                    peak_height_frac=peak_height_frac,
                    peak_distance=peak_distance,
                    top_k=top_k,
                    resize_factor=rf,
                )

                if len(peaks) > best_count:
                    best_count = len(peaks)
                    best_crop = crop
                    best_peaks = peaks

            if best_crop is None or len(best_peaks) == 0:
                print(f"[WARN] No peaks found for base '{base}', skipping.")
                continue

            # cluster_vs_gt (same meaning as clustering batch: pred_clusters == #peaks on best crop)
            gt_cat_count = self._count_gt_categories_for_base(json_path, crops)
            pred_clusters = int(best_count)
            delta = pred_clusters - int(gt_cat_count)
            relation = "equal" if delta == 0 else ("greater" if delta > 0 else "less")

            comparisons.append({
                "base": base,
                "pred_clusters": int(pred_clusters),
                "gt_categories": int(gt_cat_count),
                "relation": relation,
                "delta": int(delta),
            })

            # apply best peaks to each crop (full-res masks per crop)
            for crop in crops:
                crop_path = os.path.join(input_dir, crop)
                img_bgr = cv2.imread(crop_path)
                if img_bgr is None:
                    continue
                hsv_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

                peak_masks = self._build_peak_masks_fullres(hsv_full, best_peaks, tolerance=tolerance)
                if len(peak_masks) == 0:
                    continue

                out, cols = self._apply_peak_masks_to_crop(
                    crop_path=crop_path,
                    peak_masks=peak_masks,
                    json_path=json_path,
                    output_dir=output_dir,
                    category_colors=category_colors,
                    prefer_gt_lines=prefer_gt_lines,
                )
                all_output.extend(out)
                all_colors.extend(cols)

        # write jsons
        with open(os.path.join(output_dir, "all_data.json"), "w") as f:
            json.dump(all_output, f, indent=2)
        with open(os.path.join(output_dir, "all_colors.json"), "w") as f:
            json.dump(all_colors, f, indent=2)
        with open(os.path.join(output_dir, "cluster_vs_gt.json"), "w") as f:
            json.dump(comparisons, f, indent=2)

        # summary
        total = len(comparisons)
        less = sum(1 for r in comparisons if r.get("relation") == "less")
        equal = sum(1 for r in comparisons if r.get("relation") == "equal")
        more = sum(1 for r in comparisons if r.get("relation") == "greater")

        def pct(x, n):
            return round((x / n) * 100.0, 2) if n else 0.0

        summary = {
            "total_bases": total,
            "counts": {"less": less, "equal": equal, "greater": more},
            "percentages": {
                "less": pct(less, total),
                "equal": pct(equal, total),
                "greater": pct(more, total),
            },
        }
        with open(os.path.join(output_dir, "cluster_vs_gt_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[DONE] Saved batch results in: {output_dir}")
        return all_output, all_colors, comparisons, summary
