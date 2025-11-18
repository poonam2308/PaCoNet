import os
import re
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.pc.plot_gen.dino_feature_extractor import DinoEmbeddingWrapper


# ============================================================
# Base class: shared logic
# ============================================================
class BaseKMeansColorEvaluator:
    """
    Base class for KMeans clustering evaluation in a given feature space.

    Child classes must implement:
        - _compute_features(img_rgb: np.ndarray) -> np.ndarray of shape (N, D)
        - color_space_name (string attribute)

    Features can be anything (RGB, HSV, Lab, etc). We fit a StandardScaler
    per image, then run KMeans with K = len(category_colors) from GT JSON.
    """

    color_space_name = "BASE"

    def __init__(
        self,
        sample_size: int = 10000,
        random_state: int = 42,
        n_init: int = 10,
    ):
        self.sample_size = sample_size
        self.random_state = random_state
        self.n_init = n_init

    # ---------- Public API ----------

    def evaluate_batch(self, input_dir, json_dir, output_dir="."):
        """
        Loop over all images in input_dir, match each to its GT JSON in json_dir,
        use K = len(category_colors) from the JSON, run KMeans, compute inertia
        and silhouette score, and save one JSON summary for this color space.

        JSON file will be named: <color_space_name.lower()>_scores.json
        """
        os.makedirs(output_dir, exist_ok=True)

        per_image_results = []

        for fname in os.listdir(input_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = os.path.join(input_dir, fname)

            # Match JSON the same way as CategorySeparator.process_batch
            base_stem = re.sub(r"_crop_\d+", "", Path(fname).stem)
            json_path = os.path.join(json_dir, base_stem + ".json")

            if not os.path.exists(json_path):
                # No GT -> skip this image
                continue

            # ---- load GT to get K from category_colors ----
            with open(json_path, "r") as f:
                data = json.load(f)

            category_colors = data.get("category_colors", None)
            if not category_colors:
                # no category_colors -> cannot determine K
                continue

            k = len(category_colors)

            # ---- run evaluation for this single image ----
            metrics = self._evaluate_single_image(image_path, k)
            if metrics is None:
                # failed (e.g. not enough pixels), skip
                continue

            metrics["image"] = fname
            metrics["k"] = k
            per_image_results.append(metrics)

        # ---- compute averages ----
        avg_inertia = None
        avg_silhouette = None

        if per_image_results:
            inertias = [d["inertia"] for d in per_image_results if d["inertia"] is not None]
            silhouettes = [d["silhouette"] for d in per_image_results if d["silhouette"] is not None]

            if inertias:
                avg_inertia = float(np.mean(inertias))
            if silhouettes:
                avg_silhouette = float(np.mean(silhouettes))

        # ---- build summary dict ----
        summary = {
            "color_space": self.color_space_name,
            "n_images": len(per_image_results),
            "average_inertia": avg_inertia,
            "average_silhouette": avg_silhouette,
            "per_image": per_image_results,
        }

        # ---- save JSON ----
        out_name = f"{self.color_space_name.lower()}_scores.json"
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=4)

        print(f"[{self.color_space_name}] Saved scores to {out_path}")
        return summary

    # ---------- Internal helpers ----------

    def _evaluate_single_image(self, image_path, k):
        """
        Compute KMeans clustering for one image given K, return a dict with
        inertia and silhouette_score.
        """
        img = cv2.imread(image_path)  # BGR
        if img is None:
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        features = self._compute_features(img_rgb)  # (N, D)
        if features.size == 0:
            return None

        # Flatten to (num_pixels, D)
        pixels = features.reshape(-1, features.shape[-1]).astype(np.float32)

        # subsample for speed
        if pixels.shape[0] > self.sample_size:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(pixels.shape[0], size=self.sample_size, replace=False)
            pixels_sample = pixels[idx]
        else:
            pixels_sample = pixels

        # scale
        scaler = StandardScaler().fit(pixels_sample)
        pixels_scaled = scaler.transform(pixels_sample)

        # KMeans
        kmeans = KMeans(
            n_clusters=k,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        labels = kmeans.fit_predict(pixels_scaled)

        inertia = float(kmeans.inertia_)

        # silhouette can fail if <2 clusters actually used
        sil = None
        if len(np.unique(labels)) > 1 and pixels_scaled.shape[0] > k:
            sil = float(silhouette_score(pixels_scaled, labels, metric="euclidean"))

        return {
            "inertia": inertia,
            "silhouette": sil,
        }

    # Child classes must override this
    def _compute_features(self, img_rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ============================================================
# RGB evaluator
# ============================================================
class RGBKMeansEvaluator(BaseKMeansColorEvaluator):
    """
    Use scaled RGB (3D) as features.
    """
    color_space_name = "RGB"

    def _compute_features(self, img_rgb: np.ndarray) -> np.ndarray:
        # img_rgb already in [0,255], shape (H,W,3)
        return img_rgb.astype(np.float32)


# ============================================================
# HSV (H only) evaluator
# ============================================================
class HSVHueKMeansEvaluator(BaseKMeansColorEvaluator):
    """
    Use only the Hue channel (1D) from HSV as features.
    (This is the "HSV" case that only looks at hue.)
    """
    color_space_name = "HSV_H"

    def _compute_features(self, img_rgb: np.ndarray) -> np.ndarray:
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        h = img_hsv[:, :, 0:1]  # keep shape (H,W,1)
        return h.astype(np.float32)


# ============================================================
# Lab evaluator
# ============================================================
class LabKMeansEvaluator(BaseKMeansColorEvaluator):
    """
    Use Lab (3D) as features.
    """
    color_space_name = "LAB"

    def _compute_features(self, img_rgb: np.ndarray) -> np.ndarray:
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        return img_lab.astype(np.float32)


# ============================================================
# HSV (full H,S,V) evaluator – "HSV with all dimensions"
# ============================================================
class HSVFullKMeansEvaluator(BaseKMeansColorEvaluator):
    """
    Use full HSV (H,S,V) 3D features.
    This corresponds to your 'HSV with all dimensions' case.
    """
    color_space_name = "HSV_FULL"

    def _compute_features(self, img_rgb: np.ndarray) -> np.ndarray:
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        return img_hsv.astype(np.float32)


class EmbeddingKMeansEvaluator(BaseKMeansColorEvaluator):
    """
    KMeans evaluator on an arbitrary embedding space.
    You pass in a feature_extractor that returns an array of shape
    (N, D) or (H, W, D) given an RGB image.

    Example usage:
        dino_eval = EmbeddingKMeansEvaluator(dino_feature_extractor, sample_size=10000)
        dino_eval.evaluate_batch(input_dir, json_dir, output_dir)
    """
    color_space_name = "EMBEDDING"

    def __init__(self, feature_extractor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extractor = feature_extractor

    def _compute_features(self, img_rgb: np.ndarray) -> np.ndarray:
        feats = self.feature_extractor(img_rgb)  # should return np.ndarray
        if not isinstance(feats, np.ndarray):
            feats = np.array(feats, dtype=np.float32)
        return feats.astype(np.float32)


class DinoKMeansEvaluator(EmbeddingKMeansEvaluator):
    """
    KMeans evaluator in DINO embedding space.
    """
    color_space_name = "DINO"

    def __init__(self, model_name="vit_small_patch16_224.dino", device=None, *args, **kwargs):
        dino_extractor = DinoEmbeddingWrapper(model_name=model_name, device=device)
        super().__init__(feature_extractor=dino_extractor, *args, **kwargs)
