# clustering_category_separation_lab.py
import os
import re
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN


class LabClusteringCategorySeparator:
    """
    Clustering-based color category separation (white background), but using Lab instead of HSV hue.

    Workflow per base image:
      1) Group all its _crop_# images.
      2) Pick the "best" crop (most Lab clusters by DBSCAN on downsampled foreground pixels).
      3) Compute Lab cluster centers on that best crop.
      4) Apply those centers to *all* crops:
         - Convert each crop to Lab.
         - Assign each foreground pixel to nearest Lab center (Euclidean distance).
         - Build masks, save white-bg cutouts, assign lines.

    Foreground masking still uses HSV thresholds [S>=50, V>=50] to avoid white-ish background.

    If a JSON with `category_colors` exists for a base, cluster centers are mapped to the closest
    category in HSV (we convert Lab centers to HSV and reuse the same structure).
    Otherwise, outputs are named *_cat_1, *_cat_2, ... (stable order by cluster center hue).
    """

    # ---------- Utilities ----------
    @staticmethod
    def _json_exists(p):
        return bool(p) and os.path.exists(p)

    @staticmethod
    def _closest_category(hsv_val, category_colors):
        """
        hsv_val: (h, s, v) all in [0..1]
        category_colors: {cat: {"h":..,"s":..,"v":..}, ...}
        """
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

    def _count_gt_categories_for_base(self, json_path, crops):
        """
        Count how many GT categories are actually present (non-empty) across
        the crop_* entries for this base.
        - If lines[crop_k] is a dict: count keys with non-empty coord lists.
        - If lines[crop_k] is a list: contributes no categories (unlabeled GT).
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
            # if it's a list, there is no per-category GT to count
        return len(present)

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

    # ---------- Core clustering helpers (Lab) ----------
    def _cluster_lab(self, img_bgr, resize_factor, eps, min_samples,
                     sat_thresh=50, val_thresh=50):
        """
        Run DBSCAN on downsampled Lab of foreground pixels.
        Foreground is defined via HSV thresholds (S,V) to avoid white-ish background.

        Returns:
          cluster_centers: dict cid -> (L, a, b) as floats
          kept_ids: list of cluster ids (ints) that are not noise
        """
        # HSV for foreground mask
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        valid = (S >= sat_thresh) & (V >= val_thresh)

        # Lab for clustering
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        Lc, Ac, Bc = lab[..., 0], lab[..., 1], lab[..., 2]

        # Downsample
        L_small = cv2.resize(Lc, (0, 0), fx=resize_factor, fy=resize_factor,
                             interpolation=cv2.INTER_NEAREST)
        A_small = cv2.resize(Ac, (0, 0), fx=resize_factor, fy=resize_factor,
                             interpolation=cv2.INTER_NEAREST)
        B_small = cv2.resize(Bc, (0, 0), fx=resize_factor, fy=resize_factor,
                             interpolation=cv2.INTER_NEAREST)
        valid_small = cv2.resize(valid.astype(np.uint8), (0, 0),
                                 fx=resize_factor, fy=resize_factor,
                                 interpolation=cv2.INTER_NEAREST).astype(bool)

        lab_vals = np.stack([L_small[valid_small],
                             A_small[valid_small],
                             B_small[valid_small]], axis=1).astype(np.float32)

        if lab_vals.size == 0:
            return {}, []

        db = DBSCAN(eps=eps, min_samples=min_samples)
        lab_labels = db.fit_predict(lab_vals)
        kept = np.unique(lab_labels[lab_labels != -1])

        cluster_centers = {}
        for cid in kept:
            pts = lab_vals[lab_labels == cid]
            if pts.size == 0:
                continue
            center = pts.mean(axis=0)
            cluster_centers[int(cid)] = (float(center[0]),
                                         float(center[1]),
                                         float(center[2]))
        return cluster_centers, kept.tolist()

    def _lab_center_to_hsv01(self, center_lab):
        """
        Convert a Lab center (L,a,b in [0..255]) to HSV in [0..1] range.
        Used for category mapping and JSON color hints.
        """
        L, a, b = center_lab
        L = np.clip(L, 0, 255)
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)
        lab_pixel = np.uint8([[ [L, a, b] ]])  # shape (1,1,3)
        bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_Lab2BGR)
        hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0, 0]  # (H,S,V)

        h01 = float(hsv_pixel[0]) / 180.0
        s01 = float(hsv_pixel[1]) / 255.0
        v01 = float(hsv_pixel[2]) / 255.0
        return (h01, s01, v01)

    def _build_masks_from_centers(self, img_bgr, cluster_centers,
                                  sat_thresh=50, val_thresh=50):
        """
        Build binary masks per cluster from Lab centers.
        Steps:
          - Use HSV to find foreground pixels (S,V thresholds).
          - Convert image to Lab.
          - For every foreground pixel, assign to nearest center in Lab (Euclidean).
          - Build a mask for each cluster id.
        Returns:
          list of (cid, mask, center_hsv01) sorted by center_hsv01[0] (hue).
        """
        if not cluster_centers:
            return []

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        valid = (S >= sat_thresh) & (V >= val_thresh)

        h, w = img_bgr.shape[:2]
        lab_flat = lab.reshape(-1, 3).astype(np.float32)
        valid_flat = valid.reshape(-1)

        idx_fg = np.where(valid_flat)[0]
        if idx_fg.size == 0:
            return []

        fg_lab = lab_flat[idx_fg]  # (N_fg, 3)

        # Prepare centers in consistent order
        items = list(cluster_centers.items())  # [(cid, (L,a,b)), ...]
        cids = [cid for cid, _ in items]
        centers_arr = np.array([c for _, c in items], dtype=np.float32)  # (K,3)

        # Distances: (N_fg, K)
        diff = fg_lab[:, None, :] - centers_arr[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        nearest_idx = np.argmin(d2, axis=1)  # indices 0..K-1

        # Init masks
        masks = []
        for k, cid in enumerate(cids):
            mask_flat = np.zeros_like(valid_flat, dtype=np.uint8)
            sel = (nearest_idx == k)
            if np.any(sel):
                mask_flat[idx_fg[sel]] = 255
            mask = mask_flat.reshape(h, w)

            if np.count_nonzero(mask) == 0:
                continue

            center_lab = centers_arr[k]
            center_hsv01 = self._lab_center_to_hsv01(center_lab)
            masks.append((cid, mask, center_hsv01))

        # sort by hue (h in [0..1]) for stable naming
        masks.sort(key=lambda t: t[2][0])
        return masks

    # ---------- Single-image application ----------
    def _apply_clusters_to_image(
            self,
            crop_path,
            cluster_centers,
            json_path,
            output_dir,
            category_colors=None,
            prefer_gt_lines=True,
            sat_thresh=50,
            val_thresh=50
    ):
        """
        Apply discovered Lab cluster centers to a single crop and save white-bg cutouts.
        If `prefer_gt_lines` and per-category GT exists in json, use GT lines verbatim.
        Returns output_data, color_data.
        """
        os.makedirs(output_dir, exist_ok=True)
        img_bgr = cv2.imread(crop_path)
        if img_bgr is None:
            raise FileNotFoundError(crop_path)

        # Build masks from Lab centers
        color_masks = self._build_masks_from_centers(
            img_bgr, cluster_centers, sat_thresh=sat_thresh, val_thresh=val_thresh
        )  # list of (cid, mask, center_hsv01)

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
                    gt_cat_dict = crop_val  # keep as-is
                    for _, coords in (crop_val or {}).items():
                        if coords:
                            merged_lines.extend(coords)
                elif isinstance(crop_val, list):
                    merged_lines = crop_val or []

            # Optional category mapping hints
            if category_colors is None:
                category_colors = data.get("category_colors", None)

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
                for _, m, center_hsv01 in color_masks:
                    cat = self._closest_category(center_hsv01, category_colors)
                    cat_masks[cat] = cv2.bitwise_or(cat_masks[cat], m)
            else:
                # Original behavior: assign by midpoint-in-mask, then map to closest category
                for line in (merged_lines or []):
                    x1, y1, x2, y2 = map(int, line)
                    mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    for _, m, center_hsv01 in color_masks:
                        if 0 <= my < m.shape[0] and 0 <= mx < m.shape[1] and m[my, mx] > 0:
                            cat = self._closest_category(center_hsv01, category_colors)
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
        cat_coords = {cid: [] for cid, *_ in color_masks}
        for line in (merged_lines or []):
            x1, y1, x2, y2 = map(int, line)
            mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
            for cid, m, _ in color_masks:
                if 0 <= my < m.shape[0] and 0 <= mx < m.shape[1] and m[my, mx] > 0:
                    cat_coords[cid].append(line)
                    break

        for idx, (cid, m, center_hsv01) in enumerate(color_masks, start=1):
            if np.count_nonzero(m) == 0:
                continue
            white = np.full_like(img_bgr, 255)
            fg = cv2.bitwise_and(img_bgr, img_bgr, mask=m)
            inv = cv2.bitwise_not(m)
            out = cv2.bitwise_or(fg, cv2.bitwise_and(white, white, mask=inv))
            out_name = f"{Path(crop_path).stem}_cat_{idx}.png"
            cv2.imwrite(os.path.join(output_dir, out_name), out)
            h01, s01, v01 = center_hsv01
            output_data.append({"filename": out_name, "lines": cat_coords.get(cid, [])})
            color_data.append({
                "filename": out_name,
                "color_hsv": {
                    "h": round(h01, 2),
                    "s": round(s01, 2),
                    "v": round(v01, 2),
                }
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
        val_thresh=50,
    ):
        """
        Discover Lab clusters **on this image** and apply them to this image only.
        White background cutouts. Returns (output_data, color_data).
        """
        os.makedirs(output_dir, exist_ok=True)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(image_path)

        # Cluster Lab centers on this image
        centers, kept = self._cluster_lab(
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

        return self._apply_clusters_to_image(
            crop_path=image_path,
            cluster_centers=centers,
            json_path=json_path,
            output_dir=output_dir,
            category_colors=category_colors,
            prefer_gt_lines=True,
            sat_thresh=sat_thresh,
            val_thresh=val_thresh,
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
        prefer_gt_lines=True,
    ):
        """
        Batch mode (Lab version):
          - Group files by base (strip `_crop_#`).
          - For each base, choose the crop with most Lab clusters as the "best" crop.
          - Compute Lab centers on the best crop (downsampled foreground + DBSCAN).
          - Apply those centers to all crops in that base group.
          - Save white-bg cutouts and JSONs: all_data.json, all_colors.json, cluster_vs_gt*.json
        """
        os.makedirs(output_dir, exist_ok=True)
        comparisons = []

        # Inventory images
        files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        groups = {}
        for f in files:
            base = self._base_stem(Path(f).stem)
            groups.setdefault(base, []).append(f)

        all_output, all_colors = [], []

        for base, crops in groups.items():
            crops.sort()
            best_crop, best_centers, best_count = None, None, 0

            # 1) choose best crop by number of clusters
            for crop in crops:
                p = os.path.join(input_dir, crop)
                img = cv2.imread(p)
                if img is None:
                    continue
                rf = resize_factor_large if img.shape[1] >= width_threshold else resize_factor_small
                centers, kept = self._cluster_lab(
                    img, resize_factor=rf, eps=eps, min_samples=min_samples,
                    sat_thresh=sat_thresh, val_thresh=val_thresh
                )
                if len(kept) > best_count:
                    best_count = len(kept)
                    best_crop = crop
                    best_centers = centers

            if best_crop is None or not best_centers:
                print(f"[WARN] No valid Lab clusters for base '{base}', skipping.")
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

            # Count GT categories present across this base's crops
            gt_cat_count = self._count_gt_categories_for_base(json_path, crops)
            pred_clusters = best_count
            delta = pred_clusters - gt_cat_count
            relation = "equal" if delta == 0 else ("greater" if delta > 0 else "less")

            comparisons.append({
                "base": base,
                "pred_clusters": int(pred_clusters),
                "gt_categories": int(gt_cat_count),
                "relation": relation,
                "delta": int(delta)
            })

            # 2) apply best centers to every crop
            for crop in crops:
                crop_path = os.path.join(input_dir, crop)
                try:
                    out, cols = self._apply_clusters_to_image(
                        crop_path=crop_path,
                        cluster_centers=best_centers,
                        json_path=json_path,
                        output_dir=output_dir,
                        category_colors=category_colors,
                        prefer_gt_lines=prefer_gt_lines,
                        sat_thresh=sat_thresh,
                        val_thresh=val_thresh,
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
        with open(os.path.join(output_dir, "cluster_vs_gt.json"), "w") as f:
            json.dump(comparisons, f, indent=2)

        # ---- write summary (counts & percentages) ----
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
            }
        }

        with open(os.path.join(output_dir, "cluster_vs_gt_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[DONE] Saved Lab batch results in: {output_dir}")
        return all_output, all_colors
