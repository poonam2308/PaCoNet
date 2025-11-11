# elbo_category_separator.py
"""
ELBO-based color category separation on downsampled hue.
This class subclasses your existing ClusteringCategorySeparator and only replaces
the internal clustering step with a variational Bayesian GMM that maximizes the ELBO.

Usage (drop-in replacement):
    from elbo_category_separator import ELBOCategorySeparator

    sep = ELBOCategorySeparator()
    sep.process_single_image("path/to/img.png", json_path="path/to/img.json", output_dir="out_dir")
    # or
    sep.process_batch("input_dir", json_dir="json_dir", output_dir="out_dir")

Requirements: scikit-learn, OpenCV (cv2), numpy.
"""

from pathlib import Path
import numpy as np
import cv2
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from src.pc.plot_gen.clustering_category_separation import ClusteringCategorySeparator

class ELBOCategorySeparator(ClusteringCategorySeparator):
    """
    Same public API as ClusteringCategorySeparator, but uses a variational
    Bayes GMM on downsampled hue values. Active components are selected
    automatically via the variational prior (weights close to 0 are pruned).
    """
    def _choose_k_via_bic(self, hue_vals, k_min=1, k_max=12, sample_size=10000, random_state=0):
        """
        Choose number of clusters using BIC on a subsample of hue values.
        Returns best_k (int). Falls back to k_min if something goes wrong.
        """
        rng = np.random.default_rng(random_state)
        X = hue_vals
        if X.shape[0] > sample_size:
            idx = rng.choice(X.shape[0], size=sample_size, replace=False)
            X = X[idx]

        best_k, best_bic = k_min, np.inf
        for k in range(k_min, k_max + 1):
            try:
                gm = GaussianMixture(n_components=k, covariance_type="diag", random_state=random_state)
                gm.fit(X)
                bic = gm.bic(X)
                if bic < best_bic:
                    best_bic, best_k = bic, k
            except Exception:
                # skip impossible fits
                continue
        return int(best_k)


    def _cluster_hues(
        self,
        img_bgr,
        resize_factor,
        eps=None,                # kept for signature compatibility; ignored here
        min_samples=None,        # kept for signature compatibility; ignored here
        sat_thresh=50,
        val_thresh=50,
        n_components=8,
        weight_thresh=0.01,
        min_cluster_size=100,
        random_state=0,
        max_iter=500
    ):
        """
        Fit a BayesianGaussianMixture on the downsampled hue channel (H in [0..179])
        for moderately saturated/bright pixels only.

        Returns:
            cluster_ranges: dict[comp_id] -> (hmin, hmax, center)
            kept: list of kept component ids (stable order by center hue)
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # Keep only reasonably saturated/bright pixels to avoid white-ish background
        valid = (S >= sat_thresh) & (V >= val_thresh)
        if not np.any(valid):
            return {}, []

        # Downsample
        H_small = cv2.resize(H, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_NEAREST)
        S_small = cv2.resize(S, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_NEAREST)
        V_small = cv2.resize(V, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_NEAREST)
        valid_small = (S_small >= sat_thresh) & (V_small >= val_thresh)

        hue_vals = H_small[valid_small].astype(np.float32).reshape(-1, 1)
        if hue_vals.size == 0:
            return {}, []

        best_k = self._choose_k_via_bic(hue_vals, k_min=1, k_max=12, sample_size=15000, random_state=random_state)

        # Fit variational Bayes GMM (ELBO optimization internally)
        bgm = BayesianGaussianMixture(
            n_components=best_k,
            covariance_type="diag",
            weight_concentration_prior_type="dirichlet_process",
            max_iter=max_iter,
            random_state=random_state
        )
        bgm.fit(hue_vals)

        # Responsibilities and weights
        resp = bgm.predict_proba(hue_vals)    # (N, K)
        weights = bgm.weights_                # (K,)

        # Select active components
        active = [k for k, w in enumerate(weights) if w >= weight_thresh]
        if not active:
            return {}, []

        # Assign points by MAP component (restricted to active components)
        hard = np.argmax(resp[:, active], axis=1)     # index into 'active'
        cluster_ranges = {}
        kept_ids = []

        # Helper: handle hue wrap when computing ranges
        def hue_range_with_wrap(samples):
            s = np.sort(samples)
            # Direct interval
            direct_span = s[-1] - s[0]
            direct_center = (s[0] + s[-1]) / 2.0

            # Wrap alternative: shift values near 0 by +180 to see if span tightens
            s_shift = s.copy()
            s_shift[s_shift < 90] += 180.0
            s_shift.sort()
            wrap_span = s_shift[-1] - s_shift[0]
            wrap_center = (s_shift[0] + s_shift[-1]) / 2.0

            if wrap_span < direct_span:
                hmin_raw, hmax_raw, center_raw = s_shift[0], s_shift[-1], wrap_center
                # Map back to [0, 180)
                hmin = hmin_raw % 180.0
                hmax = hmax_raw % 180.0
                center = center_raw % 180.0
            else:
                hmin, hmax, center = s[0], s[-1], direct_center

            return float(hmin), float(hmax), float(center)

        # Build ranges per active component (using assigned samples)
        for local_k, comp_id in enumerate(active):
            comp_samples = hue_vals[hard == local_k, 0]
            if comp_samples.size < min_cluster_size:
                continue
            hmin, hmax, center = hue_range_with_wrap(comp_samples)

            # Convert bounds to ints for downstream thresholding
            cluster_ranges[int(comp_id)] = (int(np.floor(hmin)), int(np.ceil(hmax)), float(center))
            kept_ids.append(int(comp_id))

        # Sort by center hue to ensure stable ordering downstream
        kept_ids.sort(key=lambda cid: cluster_ranges[cid][2])
        cluster_ranges = {cid: cluster_ranges[cid] for cid in kept_ids}

        return cluster_ranges, kept_ids


if __name__ == "__main__":
    print("ELBOCategorySeparator ready. Import and use process_single_image/process_batch like the base class.")

