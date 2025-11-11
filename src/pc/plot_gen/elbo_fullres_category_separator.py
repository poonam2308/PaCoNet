# elbo_category_separator.py
"""
ELBO-based color category separation on FULL-RESOLUTION hue.
This subclasses your ClusteringCategorySeparator and replaces only the clustering
step with a variational Bayesian GMM (ELBO). No downsampling is performed.

Usage (drop-in):
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

# If your base class is importable directly, use:
# Import the original base class from your file
from src.pc.plot_gen.clustering_category_separation import ClusteringCategorySeparator


class ELBOFullResCategorySeparator(ClusteringCategorySeparator):
    """
    Same public API as ClusteringCategorySeparator, but uses a variational
    Bayes GMM on FULL-RESOLUTION hue values. Active components are selected
    automatically via the variational prior (tiny-weight components are pruned).
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
        resize_factor,           # kept for signature compatibility; NOT used
        eps=None,                # kept for signature compatibility; ignored
        min_samples=None,        # kept for signature compatibility; ignored
        sat_thresh=50,
        val_thresh=50,
        n_components=8,
        weight_thresh=0.01,
        min_cluster_size=100,
        random_state=0,
        max_iter=500
    ):
        """
        Fit a BayesianGaussianMixture on the FULL-RESOLUTION hue channel (H in [0..179])
        for moderately saturated/bright pixels only.

        Returns:
            cluster_ranges: dict[comp_id] -> (hmin, hmax, center)
            kept: list of kept component ids (stable order by center hue)
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # Use FULL resolution: keep only pixels that are not near-white/gray
        valid = (S >= sat_thresh) & (V >= val_thresh)
        if not np.any(valid):
            return {}, []

        # FULL set of hue values (no resize)
        hue_vals = H[valid].astype(np.float32).reshape(-1, 1)
        if hue_vals.size == 0:
            return {}, []
        best_k = self._choose_k_via_bic(hue_vals, k_min=1, k_max=12, sample_size=15000, random_state=random_state)

        # Variational Bayes GMM (optimizes the ELBO internally)
        # bgm = BayesianGaussianMixture(
        #     n_components=n_components,
        #     covariance_type="diag",
        #     weight_concentration_prior_type="dirichlet_process",
        #     max_iter=max_iter,
        #     random_state=random_state
        # )
        # bgm.fit(hue_vals)

        bgm = BayesianGaussianMixture(
            n_components=best_k,
            covariance_type="diag",
            weight_concentration_prior_type="dirichlet_process",
            max_iter=max_iter,
            random_state=random_state
        )
        bgm.fit(hue_vals)

        # Responsibilities and mixture weights
        resp = bgm.predict_proba(hue_vals)       # (N, K)
        weights = bgm.weights_                   # (K,)

        # Keep only active components by weight
        active = [k for k, w in enumerate(weights) if w >= weight_thresh]
        if not active:
            return {}, []

        # Assign each point to its MAP component among the 'active' ones
        hard = np.argmax(resp[:, active], axis=1)  # indices into 'active'
        cluster_ranges = {}
        kept_ids = []

        # Hue wrap-aware range computation (0/179 boundary)
        def hue_range_with_wrap(samples):
            s = np.sort(samples)
            # direct interval
            direct_span = s[-1] - s[0]
            direct_center = (s[0] + s[-1]) / 2.0

            # wrap alternative: shift low hues by +180
            s_shift = s.copy()
            s_shift[s_shift < 90] += 180.0
            s_shift.sort()
            wrap_span = s_shift[-1] - s_shift[0]
            wrap_center = (s_shift[0] + s_shift[-1]) / 2.0

            if wrap_span < direct_span:
                hmin_raw, hmax_raw, center_raw = s_shift[0], s_shift[-1], wrap_center
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

            cluster_ranges[int(comp_id)] = (
                int(np.floor(hmin)),
                int(np.ceil(hmax)),
                float(center),
            )
            kept_ids.append(int(comp_id))

        # Stable ordering by center hue
        kept_ids.sort(key=lambda cid: cluster_ranges[cid][2])
        cluster_ranges = {cid: cluster_ranges[cid] for cid in kept_ids}
        return cluster_ranges, kept_ids


if __name__ == "__main__":
    print("ELBOCategorySeparator ready. Import and use process_single_image/process_batch like the base class.")
