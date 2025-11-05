# clustering_category_separator.py
import os
import re
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN


class ClusteringCategorySeparator:
    """
    Clustering-based color category separation (white background), similar to CategorySeparator.
    Workflow per base image:
      1) Group all its _crop_# images.
      2) Pick the "best" crop (most clusters by DBSCAN on downsampled hue).
      3) Compute hue ranges on that best crop (min/max per cluster).
      4) Apply those ranges to *all* crops -> build masks, save white-bg cutouts, assign lines.

    If a JSON with `category_colors` exists for a base, cluster centers are mapped to the closest category.
    Otherwise, outputs are named *_cat_1, *_cat_2, ... (stable order by cluster center hue).

    Notes:
      - Foreground masking uses HSV thresholds [S>=50, V>=50] to avoid white-ish background.
      - DBSCAN parameters (eps, min_samples) and downsample factors are tunable.
      - Lines are assigned by testing the line midpoint against the mask.
    """

    # ---------- Utilities ----------
    @staticmethod
    def _json_exists(p):
        return bool(p) and os.path.exists(p)

    @staticmethod
    def _closest_category(hsv_val, category_colors):
        # hsv_val: (h in [0..1], s in [0..1], v in [0..1])
        best_cat, best_d = None, float("inf")
        for cat, ref in category_colors.items():
            rh, rs, rv = ref["h"], ref["s"], ref["v"]
            d = (hsv_val[0] - rh) ** 2 + (hsv_val[1] - rs) ** 2 + (hsv_val[2] - rv) ** 2
            if d < best_d:
                best_cat, best_d = cat, d
        return best_cat

    @staticmethod
    def _base_stem(stem: str) -> str:
        """Strip a single `_crop_#` token to get base key (order-agnostic pairing helper)."""
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

    def _load_lines(self, json_path, crop_filename):
        """
        Load line coordinates for a specific crop file. Supports two GT shapes:
          lines: {"crop_1": {"catA":[...], "catB":[...] }}  or  lines: {"crop_1":[...]}
        Returns [] if json missing/unmatched.
        """
        if not self._json_exists(json_path):
            return []

        with open(json_path, "r") as f:
            data = json.load(f)

        m = re.search(r"_crop_(\d+)", Path(crop_filename).stem)
        if not m:
            return []

        crop_key = f"crop_{m.group(1)}"
        lines_dict = (data or {}).get("lines", {})
        crop_lines = lines_dict.get(crop_key, [])

        if isinstance(crop_lines, dict):
            merged = []
            for _, coords in (crop_lines or {}).items():
                merged.extend(coords or [])
            return merged
        return crop_lines or []

    def _load_lines_structured(self, json_path, crop_filename):
        """
        Returns (per_category_dict_or_None, merged_list).
        - If lines[crop_k] is a dict: returns (that dict, merged list in dict order)
        - If it's a list: returns (None, that list)
        - If no JSON: (None, [])
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
        else:
            return None, (crop_lines or [])

    # ---------- Core clustering helpers ----------
    def _cluster_hues(self, img_bgr, resize_factor, eps, min_samples, sat_thresh=50, val_thresh=50):
        """
        Run DBSCAN on downsampled hue channel (only moderately saturated/bright pixels).
        Return: dict cluster_id -> (min_hue, max_hue, center_hue), and list of kept cluster ids.
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # Keep only reasonably saturated/bright pixels to avoid white-ish background
        valid = (S >= sat_thresh) & (V >= val_thresh)
        h_small = cv2.resize(H, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_NEAREST)
        s_small = cv2.resize(S, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_NEAREST)
        v_small = cv2.resize(V, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_NEAREST)
        valid_small = (s_small >= sat_thresh) & (v_small >= val_thresh)

        hue_vals = h_small[valid_small].reshape(-1, 1)
        if hue_vals.size == 0:
            return {}, []

        db = DBSCAN(eps=eps, min_samples=min_samples)
        lab = db.fit_predict(hue_vals)
        keep = np.unique(lab[lab != -1])

        cluster_ranges = {}
        for cid in keep:
            ch = hue_vals[lab == cid]
            if ch.size == 0:
                continue
            hmin, hmax = int(np.min(ch)), int(np.max(ch))
            center = float((hmin + hmax) / 2.0)
            cluster_ranges[int(cid)] = (hmin, hmax, center)
        return cluster_ranges, keep.tolist()

    def _build_masks_from_ranges(self, hsv_img, cluster_ranges, pad=5):
        """
        Build binary masks per cluster from (min_hue, max_hue, center).
        Returns a list of tuples (cluster_id, mask, center_hue).
        """
        masks = []
        for cid, (hmin, hmax, center) in cluster_ranges.items():
            low = np.array([max(0, hmin - pad), 50, 50], dtype=np.uint8)
            high = np.array([min(179, hmax + pad), 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv_img, low, high)
            if np.count_nonzero(mask) == 0:
                continue
            masks.append((cid, mask, center))
        # sort by increasing center hue for stable naming
        masks.sort(key=lambda t: t[2])
        return masks

    # ---------- Single-image application ----------
    def _apply_ranges_to_image(
            self,
            crop_path,
            ranges,
            json_path,
            output_dir,
            category_colors=None,
            prefer_gt_lines=True
    ):
        """
        Apply discovered hue ranges to a single crop and save white-bg cutouts.
        If `prefer_gt_lines` and per-category GT exists in json, use GT lines verbatim.
        Returns output_data, color_data.
        """
        os.makedirs(output_dir, exist_ok=True)
        img_bgr = cv2.imread(crop_path)
        if img_bgr is None:
            raise FileNotFoundError(crop_path)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Build masks from the provided cluster ranges (sorted by center hue)
        color_masks = self._build_masks_from_ranges(hsv, ranges,
                                                    pad=5)  # (cid, mask, center) :contentReference[oaicite:1]{index=1}

        # ---- Load GT lines (both shapes supported) ----
        gt_cat_dict = None  # dict: cat -> list of coords  (if JSON has per-category)
        merged_lines = []  # flat list of coords (fallback)
        if self._json_exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)

            # find crop key from filename, e.g. image_10_crop_2_*.png -> crop_2
            m = re.search(r"_crop_(\d+)", Path(crop_path).stem)
            if m:
                crop_key = f"crop_{m.group(1)}"
                crop_val = (data.get("lines") or {}).get(crop_key, [])

                if isinstance(crop_val, dict):
                    gt_cat_dict = crop_val  # keep as-is (preserve types/order)
                    # also prepare a merged list for non-category path
                    for _, coords in (crop_val or {}).items():
                        if coords:
                            merged_lines.extend(coords)
                elif isinstance(crop_val, list):
                    merged_lines = crop_val or []

            # Optional category mapping hints
            if category_colors is None:
                category_colors = data.get("category_colors", None)  # may still be None

        output_data, color_data = [], []

        # ---- With configured categories ----
        if category_colors:
            # Prepare accumulators
            cat_coords = {cat: [] for cat in category_colors}
            cat_masks = {cat: np.zeros(img_bgr.shape[:2], dtype=np.uint8) for cat in category_colors}

            if prefer_gt_lines and isinstance(gt_cat_dict, dict):
                # Use GT coordinates verbatim per category (no reassignment / casting)
                for cat in cat_coords:
                    cat_coords[cat] = gt_cat_dict.get(cat, [])
                # Still build masks for visuals: map each cluster to the nearest category
                for _, m, center in color_masks:
                    cat = self._closest_category((center / 180.0, 1, 1),
                                                 category_colors)  # :contentReference[oaicite:2]{index=2}
                    cat_masks[cat] = cv2.bitwise_or(cat_masks[cat], m)
            else:
                # Original behavior: assign by midpoint-in-mask, then map to closest category
                for line in (merged_lines or []):
                    x1, y1, x2, y2 = map(int, line)
                    mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    for _, m, center in color_masks:
                        if 0 <= my < m.shape[0] and 0 <= mx < m.shape[1] and m[my, mx] > 0:
                            cat = self._closest_category((center / 180.0, 1, 1), category_colors)
                            cat_coords.setdefault(cat, []).append(line)
                            cat_masks[cat] = cv2.bitwise_or(cat_masks[cat], m)
                            break

            # Save per-category (white bg), unchanged structure
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

        # ---- No configured categories: name outputs *_cat_# in hue order ----
        # Use merged_lines (GT union or flat) and assign by midpoint to cluster masks (stable order).
        cat_coords = {cid: [] for cid, *_ in color_masks}
        for line in (merged_lines or []):
            x1, y1, x2, y2 = map(int, line)
            mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
            for cid, m, _ in color_masks:
                if 0 <= my < m.shape[0] and 0 <= mx < m.shape[1] and m[my, mx] > 0:
                    cat_coords[cid].append(line)
                    break

        for idx, (cid, m, center) in enumerate(color_masks, start=1):
            if np.count_nonzero(m) == 0:
                continue
            white = np.full_like(img_bgr, 255)
            fg = cv2.bitwise_and(img_bgr, img_bgr, mask=m)
            inv = cv2.bitwise_not(m)
            out = cv2.bitwise_or(fg, cv2.bitwise_and(white, white, mask=inv))
            out_name = f"{Path(crop_path).stem}_cat_{idx}.png"
            cv2.imwrite(os.path.join(output_dir, out_name), out)
            output_data.append({"filename": out_name, "lines": cat_coords.get(cid, [])})
            color_data.append({
                "filename": out_name,
                "color_hsv": {"h": round(center / 180.0, 2), "s": 1, "v": 1}
            })
        return output_data, color_data

    # ---------- Public: single image ----------
    def process_single_image(
        self,
        image_path,
        json_path=None,
        output_dir=".",
        resize_factor=0.30,
        eps=5,
        min_samples=200,
        sat_thresh=50,
        val_thresh=50
    ):
        """
        Discover hue clusters **on this image** and apply them to this image only.
        White background cutouts. Returns (output_data, color_data).
        """
        os.makedirs(output_dir, exist_ok=True)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(image_path)

        # Cluster hue ranges on this image
        ranges, kept = self._cluster_hues(
            img_bgr, resize_factor=resize_factor, eps=eps, min_samples=min_samples,
            sat_thresh=sat_thresh, val_thresh=val_thresh
        )
        if not kept:
            return [], []

        # Optional categories
        category_colors = None
        if self._json_exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            category_colors = data.get("category_colors", None)

        return self._apply_ranges_to_image(
            crop_path=image_path,
            ranges=ranges,
            json_path=json_path,
            output_dir=output_dir,
            category_colors=category_colors
        )

    # ---------- Public: batch over directory (group by base; choose best crop) ----------
    def process_batch(
        self,
        input_dir,
        json_dir=None,
        output_dir=".",
        eps=5,
        min_samples=200,
        sat_thresh=50,
        val_thresh=50,
        width_threshold=200,
        resize_factor_large=0.30,
        resize_factor_small=0.325,
        prefer_gt_lines=True
    ):
        """
        Batch mode:
          - Group files by base (strip `_crop_#`).
          - For each base, choose the crop with most clusters as the "best" crop.
          - Compute hue ranges on the best crop (downsampled hue + DBSCAN).
          - Apply those ranges to all crops in that base group.
          - Save white-bg cutouts and two JSONs: all_data.json, all_colors.json
        """
        os.makedirs(output_dir, exist_ok=True)

        # Inventory images
        files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        groups = {}
        for f in files:
            base = self._base_stem(Path(f).stem)
            groups.setdefault(base, []).append(f)

        all_output, all_colors = [], []

        for base, crops in groups.items():
            crops.sort()
            # pick best resize per crop (dynamic)
            best_crop, best_ranges, best_count = None, None, 0

            # 1) choose best crop by number of clusters
            for crop in crops:
                p = os.path.join(input_dir, crop)
                img = cv2.imread(p)
                if img is None:
                    continue
                rf = resize_factor_large if img.shape[1] >= width_threshold else resize_factor_small
                ranges, kept = self._cluster_hues(
                    img, resize_factor=rf, eps=eps, min_samples=min_samples,
                    sat_thresh=sat_thresh, val_thresh=val_thresh
                )
                if len(kept) > best_count:
                    best_count = len(kept)
                    best_crop = crop
                    best_ranges = ranges
                    best_rf = rf

            if best_crop is None or not best_ranges:
                print(f"[WARN] No valid clusters for base '{base}', skipping.")
                continue

            # Optional per-base JSON
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

            print(f"[INFO] Base '{base}': using best crop '{best_crop}' with {best_count} clusters.")

            # 2) apply best crop ranges to every crop
            for crop in crops:
                crop_path = os.path.join(input_dir, crop)
                try:
                    out, cols = self._apply_ranges_to_image(
                        crop_path=crop_path,
                        ranges=best_ranges,
                        json_path=json_path,
                        output_dir=output_dir,
                        category_colors=category_colors,
                        prefer_gt_lines=prefer_gt_lines
                    )
                    all_output.extend(out)
                    all_colors.extend(cols)
                except Exception as e:
                    print(f"[WARN] Skipping {crop_path}: {e}")

        # Save consolidated JSONs
        with open(os.path.join(output_dir, "all_data.json"), "w") as f:
            json.dump(all_output, f, indent=2)
        with open(os.path.join(output_dir, "all_colors.json"), "w") as f:
            json.dump(all_colors, f, indent=2)

        print(f"[DONE] Saved batch results in: {output_dir}")
        return all_output, all_colors
