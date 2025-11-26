from scipy.signal import find_peaks
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import os
from pathlib import Path
import json
import re


class PeakClusteringCategorySeparator:
    """
    Peak-aware clustering + category separation (white background).

      1) Build a hue histogram from saturation/value-filtered pixels.
      2) Detect peaks on the histogram.
      3) Optionally keep only the top-K strongest peaks.
      4) Keep only pixels whose hue lies within +/- peak_tol of any selected peak.
      5) Run DBSCAN on this filtered hue set.
      6) Apply resulting hue ranges to the image (white background, with lines).

    This is a standalone version (no inheritance from ClusteringCategorySeparator).
    """

    # ---------- Small utilities ----------

    @staticmethod
    def _json_exists(p):
        return bool(p) and os.path.exists(p)

    @staticmethod
    def _closest_category(hsv_val, category_colors):
        """
        hsv_val: (h in [0..1], s in [0..1], v in [0..1])
        category_colors: {cat: {"h":..., "s":..., "v":...}, ...}
        """
        best_cat, best_d = None, float("inf")
        for cat, ref in category_colors.items():
            rh, rs, rv = ref["h"], ref["s"], ref["v"]
            d = (hsv_val[0] - rh) ** 2 + (hsv_val[1] - rs) ** 2 + (hsv_val[2] - rv) ** 2
            if d < best_d:
                best_cat, best_d = cat, d
        return best_cat

    def _build_masks_from_ranges(self, hsv_img, cluster_ranges, pad=5):
        """
        Build binary masks per cluster from (min_hue, max_hue, center).

        Parameters
        ----------
        hsv_img : np.ndarray (H,W,3) HSV image
        cluster_ranges : dict[int] -> (hmin, hmax, center)
        pad : int
            Extra margin in hue around [hmin, hmax].

        Returns
        -------
        masks : list of (cluster_id, mask, center_hue)
            Sorted by center_hue ascending for stable naming.
        """
        masks = []
        for cid, (hmin, hmax, center) in cluster_ranges.items():
            low = np.array([max(0, hmin - pad), 50, 50], dtype=np.uint8)
            high = np.array([min(179, hmax + pad), 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv_img, low, high)
            if np.count_nonzero(mask) == 0:
                continue
            masks.append((cid, mask, center))
        masks.sort(key=lambda t: t[2])  # sort by center hue
        return masks

    def _count_gt_categories_for_crop(self, json_path, crop_filename):
        """
        Count how many GT categories are present for a single crop file.

        - If lines[crop_k] is a dict: count keys with non-empty coord lists.
        - If lines[crop_k] is a list: contributes 0 (unlabeled GT).
        """
        if not self._json_exists(json_path):
            return 0

        with open(json_path, "r") as f:
            data = json.load(f)

        m = re.search(r"_crop_(\d+)", Path(crop_filename).stem)
        if not m:
            return 0

        crop_key = f"crop_{m.group(1)}"
        lines_root = (data.get("lines") or {})
        v = lines_root.get(crop_key, [])

        if isinstance(v, dict):
            count = 0
            for _, coords in v.items():
                if coords and len(coords) > 0:
                    count += 1
            return count
        # if it's a list, there is no per-category GT to count
        return 0

    def _apply_ranges_to_image(
        self,
        crop_path,
        ranges,
        json_path,
        output_dir,
        category_colors=None,
        prefer_gt_lines=True,
    ):
        """
        Apply discovered hue ranges to a single crop and save white-bg cutouts.

        - If `category_colors` is provided (or found in json), outputs are per-category.
        - Otherwise, outputs are named *_cat_1, *_cat_2, ... (ordered by hue).
        - If `prefer_gt_lines` and per-category GT exists in json, use GT lines verbatim.
        - Otherwise, lines are assigned by midpoint-in-mask.

        Returns
        -------
        output_data : list[{"filename": str, "lines": [...] }]
        color_data  : list[{"filename": str, "color_hsv": {...}}]
        """
        os.makedirs(output_dir, exist_ok=True)
        img_bgr = cv2.imread(crop_path)
        if img_bgr is None:
            raise FileNotFoundError(crop_path)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Build masks from the provided cluster ranges (sorted by center hue)
        color_masks = self._build_masks_from_ranges(hsv, ranges, pad=5)

        # ---- Load GT lines (both shapes supported) ----
        gt_cat_dict = None  # dict: cat -> list of coords  (if JSON has per-category)
        merged_lines = []   # flat list of coords (fallback)

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
                    cat = self._closest_category((center / 180.0, 1, 1), category_colors)
                    cat_masks[cat] = cv2.bitwise_or(cat_masks[cat], m)
            else:
                # Assign by midpoint-in-mask, then map to closest category
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

                output_data.append({
                    "filename": out_name,
                    "lines": cat_coords.get(cat, []),
                })
                color_data.append({
                    "filename": out_name,
                    "color_hsv": category_colors[cat],
                })
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

            output_data.append({
                "filename": out_name,
                "lines": cat_coords.get(cid, []),
            })
            color_data.append({
                "filename": out_name,
                "color_hsv": {
                    "h": round(center / 180.0, 2),
                    "s": 1,
                    "v": 1,
                }
            })
        return output_data, color_data

    # ---------- Peak-aware hue clustering ----------

    def _cluster_hues(
        self,
        img_bgr,
        resize_factor,
        eps,
        min_samples,
        sat_thresh=50,
        val_thresh=50,
        use_peaks=True,
        top_k=None,
        peak_height_frac=0.05,
        peak_distance=5,
        peak_tol=10,
    ):
        """
        Downsampled hue clustering with optional peak pre-filtering.

        Returns
        -------
        cluster_ranges : dict[int] -> (min_hue, max_hue, center_hue)
        kept_clusters  : list[int]
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # Downsample
        h_small = cv2.resize(
            H, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_NEAREST
        )
        s_small = cv2.resize(
            S, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_NEAREST
        )
        v_small = cv2.resize(
            V, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_NEAREST
        )

        valid_small = (s_small >= sat_thresh) & (v_small >= val_thresh)
        hue_vals = h_small[valid_small].reshape(-1, 1)

        if hue_vals.size == 0:
            return {}, []

        # ---- Peak pre-filtering ----
        if use_peaks:
            # Histogram over all valid downsampled hues (OpenCV hue range [0, 180))
            hue_flat = hue_vals.flatten()
            hist, _ = np.histogram(hue_flat, bins=180, range=(0, 180))

            if hist.max() > 0:
                # Detect peaks
                peaks, props = find_peaks(
                    hist,
                    height=hist.max() * peak_height_frac,
                    distance=int(peak_distance),
                )

                # Optionally keep only strongest top_k
                if top_k is not None and len(peaks) > 0:
                    try:
                        k = int(top_k)
                    except Exception:
                        k = None
                    if k is not None and k > 0 and len(peaks) > k:
                        order = np.argsort(hist[peaks])[::-1]
                        peaks = peaks[order[:k]]

                if len(peaks) > 0:
                    # Keep only hues within +/- peak_tol of *any* selected peak
                    keep_mask = np.zeros_like(hue_flat, dtype=bool)
                    for p in peaks:
                        low = max(0, p - peak_tol)
                        high = min(179, p + peak_tol)
                        keep_mask |= (hue_flat >= low) & (hue_flat <= high)

                    hue_flat = hue_flat[keep_mask]
                    hue_vals = hue_flat.reshape(-1, 1)

        # After peak filtering we might lose everything
        if hue_vals.size == 0:
            return {}, []

        # ---- DBSCAN on (possibly) peak-filtered hues ----
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(hue_vals)

        kept_clusters = np.unique(labels[labels != -1])
        cluster_ranges = {}

        for cid in kept_clusters:
            ch = hue_vals[labels == cid]
            if ch.size == 0:
                continue
            hmin, hmax = int(np.min(ch)), int(np.max(ch))
            center = float((hmin + hmax) / 2.0)
            cluster_ranges[int(cid)] = (hmin, hmax, center)

        return cluster_ranges, kept_clusters.tolist()

    # ---------- Public API ----------

    def process_single_image_with_peaks(
        self,
        image_path,
        json_path=None,
        output_dir=".",
        resize_factor=0.30,
        eps=5,
        min_samples=200,
        sat_thresh=50,
        val_thresh=50,
        top_k=None,
        peak_height_frac=0.05,
        peak_distance=5,
        peak_tol=10,
        prefer_gt_lines=True,
    ):
        """
        Run the hybrid 'peaks -> DBSCAN' pipeline on a single image.

        - Detect hue peaks on downsampled hue.
        - Filter hue samples to lie near those peaks.
        - Run DBSCAN on filtered samples.
        - Apply resulting hue ranges to this image only (white background).

        Returns
        -------
        output_data : list[dict]
        color_data  : list[dict]
        """
        os.makedirs(output_dir, exist_ok=True)

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Discover ranges with peak-aware clustering
        ranges, kept = self._cluster_hues(
            img_bgr,
            resize_factor=resize_factor,
            eps=eps,
            min_samples=min_samples,
            sat_thresh=sat_thresh,
            val_thresh=val_thresh,
            use_peaks=True,
            top_k=top_k,
            peak_height_frac=peak_height_frac,
            peak_distance=peak_distance,
            peak_tol=peak_tol,
        )

        if not kept:
            print(f"[WARN] No clusters found (after peak filtering) for '{image_path}'.")
            return [], []

        # Apply to this image
        output_data, color_data = self._apply_ranges_to_image(
            crop_path=image_path,
            ranges=ranges,
            json_path=json_path,
            output_dir=output_dir,
            category_colors=None,        # let it load from json if present
            prefer_gt_lines=prefer_gt_lines,
        )
        return output_data, color_data

    def process_batch_with_peaks(
            self,
            input_dir,
            json_dir=None,
            output_dir=".",
            resize_factor=0.30,
            eps=5,
            min_samples=200,
            sat_thresh=50,
            val_thresh=50,
            top_k=None,
            peak_height_frac=0.05,
            peak_distance=5,
            peak_tol=10,
            prefer_gt_lines=True,
            save_per_file=False,
    ):
        """
        Simple batch mode using the hybrid peaks+DBSCAN approach.

        For every image in input_dir:
          - optional matching JSON from json_dir (same base stem),
          - run peak-aware clustering (peaks -> DBSCAN),
          - apply ranges to the image,
          - append results to all_data/all_colors,
          - optionally write per-file JSON,
          - record cluster-vs-GT evaluation per image.
        """
        os.makedirs(output_dir, exist_ok=True)

        files = [
            f for f in os.listdir(input_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        all_output, all_colors = [], []
        evaluations = []

        for fname in sorted(files):
            img_path = os.path.join(input_dir, fname)
            json_path = None

            # Match JSON by base (strip `_crop_#` like in the other class)
            if json_dir is not None:
                stem = Path(fname).stem
                base = re.sub(r"_crop_\d+", "", stem)
                cand = os.path.join(json_dir, base + ".json")
                if os.path.exists(cand):
                    json_path = cand

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"[WARN] Could not read image '{img_path}', skipping.")
                continue

            # --- peak-aware clustering (for this image) ---
            ranges, kept = self._cluster_hues(
                img_bgr,
                resize_factor=resize_factor,
                eps=eps,
                min_samples=min_samples,
                sat_thresh=sat_thresh,
                val_thresh=val_thresh,
                use_peaks=True,
                top_k=top_k,
                peak_height_frac=peak_height_frac,
                peak_distance=peak_distance,
                peak_tol=peak_tol,
            )

            pred_clusters = len(kept)
            gt_categories = self._count_gt_categories_for_crop(json_path, fname)
            delta = pred_clusters - gt_categories
            relation = "equal" if delta == 0 else ("greater" if delta > 0 else "less")

            evaluations.append({
                "file": fname,
                "pred_clusters": int(pred_clusters),
                "gt_categories": int(gt_categories),
                "relation": relation,
                "delta": int(delta),
            })

            if not kept:
                print(f"[WARN] No clusters found (after peak filtering) for '{img_path}'.")
                # nothing to apply; continue to next file
                continue

            # --- apply ranges to this image (white background, with lines) ---
            out_data, col_data = self._apply_ranges_to_image(
                crop_path=img_path,
                ranges=ranges,
                json_path=json_path,
                output_dir=output_dir,
                category_colors=None,  # let it load from json if present
                prefer_gt_lines=prefer_gt_lines,
            )

            all_output.extend(out_data)
            all_colors.extend(col_data)

            if save_per_file:
                stem = Path(fname).stem
                with open(os.path.join(output_dir, f"{stem}_data.json"), "w") as f:
                    json.dump(out_data, f, indent=2)
                with open(os.path.join(output_dir, f"{stem}_colors.json"), "w") as f:
                    json.dump(col_data, f, indent=2)

        # Global summaries for cutouts/colors
        with open(os.path.join(output_dir, "all_data_peaks.json"), "w") as f:
            json.dump(all_output, f, indent=2)
        with open(os.path.join(output_dir, "all_colors_peaks.json"), "w") as f:
            json.dump(all_colors, f, indent=2)

        # --- Save cluster vs GT evaluation (per image) ---
        with open(os.path.join(output_dir, "peaks_cluster_vs_gt.json"), "w") as f:
            json.dump(evaluations, f, indent=2)

        # Summary counts & percentages
        total = len(evaluations)
        less = sum(1 for r in evaluations if r.get("relation") == "less")
        equal = sum(1 for r in evaluations if r.get("relation") == "equal")
        greater = sum(1 for r in evaluations if r.get("relation") == "greater")

        def pct(x, n):
            return round((x / n) * 100.0, 2) if n else 0.0

        summary = {
            "total_images": total,
            "counts": {"less": less, "equal": equal, "greater": greater},
            "percentages": {
                "less": pct(less, total),
                "equal": pct(equal, total),
                "greater": pct(greater, total),
            },
        }

        with open(os.path.join(output_dir, "peaks_cluster_vs_gt_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[DONE] Peak+DBSCAN batch saved in: {output_dir}")
        return all_output, all_colors
