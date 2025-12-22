import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


@dataclass
class SeparationResult:
    image_path: str
    masks: List[np.ndarray]          # list of uint8 masks (0/255)
    centers_ab: np.ndarray           # (k,2) cluster centers in Lab a/b
    counts: List[int]                # number of pixels in each mask


class PlotCategoryColorSeparator:
    """
    Robust category-color detection + separation for plot images (e.g., parallel coordinates).
    Strategy:
      1) Choose a reference crop (cleanest = most colored pixels).
      2) Learn K color clusters (Lab a/b) from the reference.
      3) For each crop, assign pixels to nearest learned center => consistent categories.
    """

    def __init__(
        self,
        sat_thresh: int = 40,
        val_thresh: int = 40,
        drop_red: bool = True,
        red_hue_low: int = 8,     # OpenCV hue 0..179
        red_hue_high: int = 172,
        red_sat_min: int = 80,
        red_val_min: int = 60,
        min_pixels_per_cat: int = 150,   # to suppress tiny junk masks
        morph_open: int = 0,             # set 1..3 if you want to remove speckles
        morph_close: int = 0,            # set 1..3 if you want to connect broken lines
    ):
        self.sat_thresh = sat_thresh
        self.val_thresh = val_thresh

        self.drop_red = drop_red
        self.red_hue_low = red_hue_low
        self.red_hue_high = red_hue_high
        self.red_sat_min = red_sat_min
        self.red_val_min = red_val_min

        self.min_pixels_per_cat = min_pixels_per_cat
        self.morph_open = morph_open
        self.morph_close = morph_close

        # Learned from reference crop:
        self.centers_ab: Optional[np.ndarray] = None  # shape (k,2)

    # ---------- IO helpers ----------
    @staticmethod
    def _read_bgr(image_path: str) -> np.ndarray:
        pil = Image.open(image_path).convert("RGB")
        rgb = np.array(pil)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    @staticmethod
    def _list_images(folder: str) -> List[str]:
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(exts)
        ]
        return sorted(files)

    # ---------- Pixel filtering ----------
    def _colored_pixel_mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        colored = (s > self.sat_thresh) & (v > self.val_thresh)

        if self.drop_red:
            # Red wraps around hue=0 in HSV; we exclude red separator lines
            red = (((h < self.red_hue_low) | (h > self.red_hue_high)) &
                   (s > self.red_sat_min) & (v > self.red_val_min))
            colored = colored & (~red)

        return colored

    def _count_colored_pixels(self, bgr: np.ndarray) -> int:
        m = self._colored_pixel_mask(bgr)
        return int(np.count_nonzero(m))

    # ---------- Learning K from reference ----------
    def _kmeans_ab(self, bgr: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          labels_full: (H,W) int32 labels in [0..k-1], -1 for ignored pixels
          centers_ab: (k,2) float32
          colored_mask: (H,W) bool
        """
        colored = self._colored_pixel_mask(bgr)
        if np.count_nonzero(colored) < max(300, k * 50):
            raise ValueError("Not enough colored pixels to cluster reliably.")

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        ab = lab[:, :, 1:3].reshape(-1, 2).astype(np.float32)

        idx = np.where(colored.reshape(-1))[0]
        samples = ab[idx]

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
        _, labels, centers = cv2.kmeans(
            samples, k, None, criteria, 6, cv2.KMEANS_PP_CENTERS
        )

        labels_full = np.full((bgr.shape[0] * bgr.shape[1],), -1, dtype=np.int32)
        labels_full[idx] = labels.flatten().astype(np.int32)
        labels_full = labels_full.reshape(bgr.shape[:2])

        return labels_full, centers.astype(np.float32), colored

    @staticmethod
    def _silhouette_like_score(samples: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
        """
        Lightweight scoring: ratio of between-center distances to within-cluster distances.
        Higher is better.
        """
        # within (mean distance to center)
        d = np.linalg.norm(samples - centers[labels], axis=1)
        within = float(np.mean(d) + 1e-6)

        # between (mean pairwise center distance)
        if centers.shape[0] < 2:
            return 0.0
        cd = centers
        # pairwise distances without heavy scipy
        sum_dist = 0.0
        cnt = 0
        for i in range(len(cd)):
            for j in range(i + 1, len(cd)):
                sum_dist += float(np.linalg.norm(cd[i] - cd[j]))
                cnt += 1
        between = (sum_dist / max(cnt, 1)) + 1e-6

        return between / within

    def infer_k_from_reference(
        self,
        reference_bgr: np.ndarray,
        k_min: int = 2,
        k_max: int = 6,
        max_samples: int = 8000,
        seed: int = 0
    ) -> int:
        """
        Automatically pick K by trying k_min..k_max and using a simple separation score.
        """
        rng = np.random.default_rng(seed)
        colored = self._colored_pixel_mask(reference_bgr)
        lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB)
        ab = lab[:, :, 1:3].reshape(-1, 2).astype(np.float32)
        idx = np.where(colored.reshape(-1))[0]
        if len(idx) < 500:
            raise ValueError("Reference has too few colored pixels to infer K.")

        # subsample for speed/stability
        if len(idx) > max_samples:
            idx = rng.choice(idx, size=max_samples, replace=False)
        samples = ab[idx]

        best_k = k_min
        best_score = -1.0

        for k in range(k_min, k_max + 1):
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
            _, labels, centers = cv2.kmeans(samples, k, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
            labels = labels.flatten().astype(np.int32)
            centers = centers.astype(np.float32)

            score = self._silhouette_like_score(samples, labels, centers)
            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    def fit_from_crops(
        self,
        crop_paths_or_dir: Union[List[str], str],
        k: Optional[int] = None,
        k_min: int = 2,
        k_max: int = 6
    ) -> int:
        """
        Choose cleanest crop, learn centers_ab from it. Returns the chosen K.
        """
        if isinstance(crop_paths_or_dir, (str, os.PathLike)) and os.path.isdir(crop_paths_or_dir):
            crop_paths = self._list_images(str(crop_paths_or_dir))
        else:
            crop_paths = list(crop_paths_or_dir)

        if not crop_paths:
            raise ValueError("No crop images provided.")

        # choose cleanest crop
        bgrs = [(p, self._read_bgr(p)) for p in crop_paths]
        counts = [self._count_colored_pixels(bgr) for _, bgr in bgrs]
        ref_i = int(np.argmax(counts))
        ref_path, ref_bgr = bgrs[ref_i]

        # infer K if not provided
        if k is None:
            k = self.infer_k_from_reference(ref_bgr, k_min=k_min, k_max=k_max)

        # learn centers from reference
        _, centers, _ = self._kmeans_ab(ref_bgr, k)
        self.centers_ab = centers

        return k

    # ---------- Apply learned colors to any crop ----------
    def _assign_to_centers(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns labels_full (H,W) in [0..k-1] or -1 for ignored pixels, and colored mask.
        """
        if self.centers_ab is None:
            raise RuntimeError("Call fit_from_crops(...) first (or set centers_ab).")

        colored = self._colored_pixel_mask(bgr)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        ab = lab[:, :, 1:3].astype(np.float32)  # (H,W,2)

        H, W = bgr.shape[:2]
        labels_full = np.full((H, W), -1, dtype=np.int32)

        ys, xs = np.where(colored)
        if len(xs) == 0:
            return labels_full, colored

        pts = ab[ys, xs]  # (N,2)
        # compute nearest center (broadcast)
        # distances: (N,k)
        d = np.linalg.norm(pts[:, None, :] - self.centers_ab[None, :, :], axis=2)
        labels = np.argmin(d, axis=1).astype(np.int32)

        labels_full[ys, xs] = labels
        return labels_full, colored

    def _postprocess_mask(self, mask255: np.ndarray) -> np.ndarray:
        m = mask255.copy()

        if self.morph_open > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*self.morph_open+1, 2*self.morph_open+1))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

        if self.morph_close > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*self.morph_close+1, 2*self.morph_close+1))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

        return m

    def separate(self, image_path: str) -> SeparationResult:
        bgr = self._read_bgr(image_path)
        labels_full, _ = self._assign_to_centers(bgr)

        k = self.centers_ab.shape[0]
        masks = []
        counts = []

        for ci in range(k):
            m = (labels_full == ci).astype(np.uint8) * 255
            m = self._postprocess_mask(m)
            c = int(np.count_nonzero(m))
            masks.append(m)
            counts.append(c)

        # suppress tiny junk masks by zeroing them (keeps list length consistent)
        for i, c in enumerate(counts):
            if c < self.min_pixels_per_cat:
                masks[i][:] = 0
                counts[i] = 0

        return SeparationResult(
            image_path=image_path,
            masks=masks,
            centers_ab=self.centers_ab.copy(),
            counts=counts
        )

    def separate_group_to_folder(
        self,
        crop_paths_or_dir: Union[List[str], str],
        output_dir: str,
        prefix: str = "",
    ) -> List[SeparationResult]:
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(crop_paths_or_dir, (str, os.PathLike)) and os.path.isdir(crop_paths_or_dir):
            crop_paths = self._list_images(str(crop_paths_or_dir))
        else:
            crop_paths = list(crop_paths_or_dir)

        results = []
        for p in crop_paths:
            res = self.separate(p)
            results.append(res)

            base = os.path.splitext(os.path.basename(p))[0]
            for i, m in enumerate(res.masks, start=1):
                out_path = os.path.join(output_dir, f"{prefix}{base}_cat{i}.png")
                cv2.imwrite(out_path, m)

        return results
