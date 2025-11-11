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
            resize_factor,  # kept for signature compatibility; NOT used
            eps=None,  # kept for signature compatibility; ignored
            min_samples=None,  # kept for signature compatibility; ignored
            sat_thresh=50,
            val_thresh=50,
            n_components=8,  # unused when BIC chooses K; kept for compatibility
            weight_thresh=0.01,
            min_cluster_size=100,
            random_state=0,
            max_iter=500
    ):
        """
        Full-resolution ELBO clustering on hue with strong numerical safeguards.
        - float64 data for stability
        - BIC to choose K upper bound
        - reg_covar, init fallback, covariance_type fallback
        - graceful early-exit if hue has near-zero variance
        """

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # Use FULL resolution: filter out near-white/gray pixels
        valid = (S >= sat_thresh) & (V >= val_thresh)
        if not np.any(valid):
            return {}, []

        # Float64 (not float32) to avoid precision issues inside sklearn mixtures
        hue_vals = H[valid].astype(np.float64).reshape(-1, 1)
        if hue_vals.size == 0:
            return {}, []

        # If data is (almost) constant, return a single tight range
        var = np.var(hue_vals)
        unique_count = np.unique(hue_vals).size
        if var < 1e-6 or unique_count <= 2:
            hmin = int(np.floor(np.min(hue_vals)))
            hmax = int(np.ceil(np.max(hue_vals)))
            center = float((hmin + hmax) / 2.0)
            return {0: (hmin, hmax, center)}, [0]

        # --- Choose K via BIC on a subsample (fast & stable) ---
        # Clamp to data complexity to avoid overparameterization
        best_k = self._choose_k_via_bic(
            hue_vals,
            k_min=1,
            k_max=min(12, max(1, unique_count)),  # never ask for more comps than unique hues
            sample_size=15000,
            random_state=random_state
        )
        best_k = max(1, min(int(best_k), unique_count))

        # Helper: hue wrap-aware range (0..179)
        def hue_range_with_wrap(samples: np.ndarray):
            s = np.sort(samples)
            direct_span = s[-1] - s[0]
            direct_center = (s[0] + s[-1]) / 2.0

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

        # --- Try robust Bayesian fit first (ELBO) ---
        def fit_bayesian(init_params="kmeans", covariance_type="diag", reg_covar=1e-3):
            bgm = BayesianGaussianMixture(
                n_components=best_k,
                covariance_type=covariance_type,
                weight_concentration_prior_type="dirichlet_process",
                init_params=init_params,
                reg_covar=reg_covar,  # key: prevent collapsed covariance
                tol=1e-4,
                max_iter=max_iter,
                random_state=random_state
            )
            return bgm.fit(hue_vals), "bayesian"

        model, mode = None, None
        try:
            model, mode = fit_bayesian(init_params="kmeans", covariance_type="diag", reg_covar=1e-3)
        except Exception:
            try:
                # Add more regularization and random init
                model, mode = fit_bayesian(init_params="random", covariance_type="diag", reg_covar=1e-2)
            except Exception:
                try:
                    # 1D: spherical is equivalent to diag but can be numerically friendlier
                    model, mode = fit_bayesian(init_params="random", covariance_type="spherical", reg_covar=1e-2)
                except Exception:
                    # Final fallback: plain GMM with strong regularization
                    gm = GaussianMixture(
                        n_components=best_k,
                        covariance_type="diag",
                        init_params="kmeans",
                        reg_covar=1e-2,
                        tol=1e-4,
                        max_iter=max_iter,
                        random_state=random_state
                    )
                    model = gm.fit(hue_vals)
                    mode = "gmm"

        # Responsibilities + weights (available in both Bayesian & vanilla GMM)
        resp = model.predict_proba(hue_vals)
        weights = model.weights_

        # Keep only active components by weight (prune tiny ones)
        active = [k for k, w in enumerate(weights) if w >= weight_thresh]
        if not active:
            # If everything got pruned, keep the single largest-weight component
            active = [int(np.argmax(weights))]

        # Assign each point to its MAP component among the active ones
        hard = np.argmax(resp[:, active], axis=1)  # indices into 'active'
        cluster_ranges, kept_ids = {}, []

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

        # If everything was too small, at least return the global range
        if not kept_ids:
            hmin = int(np.floor(np.min(hue_vals)))
            hmax = int(np.ceil(np.max(hue_vals)))
            center = float((hmin + hmax) / 2.0)
            cluster_ranges = {0: (hmin, hmax, center)}
            kept_ids = [0]

        # Stable ordering by center hue
        kept_ids.sort(key=lambda cid: cluster_ranges[cid][2])
        cluster_ranges = {cid: cluster_ranges[cid] for cid in kept_ids}
        return cluster_ranges, kept_ids


if __name__ == "__main__":
    print("ELBOCategorySeparator ready. Import and use process_single_image/process_batch like the base class.")
