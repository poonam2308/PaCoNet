import numpy as np


class LineSegmentSAPMetric:
    """
    Compute structural Average Precision (sAP) for line segments.

    - You feed it per-image:
        * pred_lines : [N_pred, 2, 2]  (y, x) endpoints in the same coord system as GT
        * pred_scores: [N_pred]        confidence scores
        * gt_lines   : [N_gt, 2, 2]    (y, x) endpoints

    - It aggregates true/false positives across the whole dataset for one or more
      distance thresholds, then computes AP in the same style as eval-sAP.py.

    Typical usage:

        metric = LineSegmentSAPMetric(thresholds=(5.0, 10.0, 15.0))

        for each image:
            metric.add_image(pred_lines, pred_scores, gt_lines)

        sap_dict = metric.compute_sap()
        # e.g. sap_dict[5.0] = sAP5, sap_dict[10.0] = sAP10, ...

    You can then print 100 * sAP for percentage if you like.
    """

    def __init__(self, thresholds=(5.0, 10.0, 15.0)):
        self.thresholds = tuple(float(t) for t in thresholds)
        self.reset()

    def reset(self):
        # Store per-threshold TP/FP vectors and GT counts
        self._tp = {t: [] for t in self.thresholds}
        self._fp = {t: [] for t in self.thresholds}
        self._n_gt = {t: 0 for t in self.thresholds}

        # Scores are shared across thresholds (same predictions)
        self._scores = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_image(self, pred_lines, pred_scores, gt_lines):
        """
        Add predictions and GT for a single image.

        Parameters
        ----------
        pred_lines : array-like, shape [N_pred, 2, 2]
        pred_scores : array-like, shape [N_pred]
        gt_lines : array-like, shape [N_gt, 2, 2]
        """
        pred_lines = np.asarray(pred_lines, dtype=np.float32)
        pred_scores = np.asarray(pred_scores, dtype=np.float32)
        gt_lines = np.asarray(gt_lines, dtype=np.float32)

        if pred_lines.ndim != 3 or pred_lines.shape[1:] != (2, 2):
            raise ValueError(
                f"pred_lines must be [N_pred, 2, 2], got shape {pred_lines.shape}"
            )
        if gt_lines.ndim != 3 or gt_lines.shape[1:] != (2, 2):
            raise ValueError(
                f"gt_lines must be [N_gt, 2, 2], got shape {gt_lines.shape}"
            )
        if pred_scores.shape[0] != pred_lines.shape[0]:
            raise ValueError(
                f"pred_scores length {pred_scores.shape[0]} does not match "
                f"pred_lines {pred_lines.shape[0]}"
            )

        N_pred = pred_lines.shape[0]
        N_gt = gt_lines.shape[0]

        # For each threshold, compute tp/fp for these predictions
        for tau in self.thresholds:
            tp_vec, fp_vec, n_gt = self._match_lines(pred_lines, gt_lines, tau=tau)
            # tp_vec / fp_vec are length N_pred
            if tp_vec.shape[0] != N_pred or fp_vec.shape[0] != N_pred:
                raise RuntimeError("Internal error: TP/FP length mismatch with predictions.")

            self._tp[tau].append(tp_vec)
            self._fp[tau].append(fp_vec)
            self._n_gt[tau] += n_gt

        # Scores are shared across thresholds
        if N_pred > 0:
            self._scores.append(pred_scores)

    def compute_sap(self):
        """
        Compute sAP for each threshold specified at construction.

        Returns
        -------
        dict: {threshold: ap_value}
            ap_value is in [0, 1] (multiply by 100 to get percentage).
            If no valid GT or predictions are present, value may be np.nan.
        """
        results = {}

        if not self._scores:
            # No predictions at all
            for tau in self.thresholds:
                results[tau] = np.nan
            return results

        # Concatenate scores only once (shared across thresholds)
        scores_all = np.concatenate(self._scores)

        for tau in self.thresholds:
            if not self._tp[tau]:
                results[tau] = np.nan
                continue

            tp_all = np.concatenate(self._tp[tau])  # 0/1 vector per prediction
            fp_all = np.concatenate(self._fp[tau])  # 0/1 vector per prediction
            n_gt = self._n_gt[tau]

            if n_gt == 0 or scores_all.size == 0:
                # No ground truth for this threshold or no predictions
                results[tau] = np.nan
                continue

            # Sort predictions by score (descending)
            order = np.argsort(-scores_all)
            tp_sorted = tp_all[order]
            fp_sorted = fp_all[order]

            # Cumulative sum → TP/FP curves normalized by number of GT lines
            tp_cum = np.cumsum(tp_sorted).astype(np.float32) / float(n_gt)
            fp_cum = np.cumsum(fp_sorted).astype(np.float32) / float(n_gt)

            # Recall is TP / (TP+FN) = tp_cum (since denominator is n_gt)
            recall = tp_cum
            # Precision = TP / (TP + FP)
            precision = tp_cum / (tp_cum + fp_cum + 1e-8)

            ap = self._average_precision(recall, precision)
            results[tau] = ap

        return results

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _match_lines(pred_lines, gt_lines, tau):
        """
        One-to-one greedy matching between predicted and GT line segments.

        Distance metric: average endpoint error (like your current match_lines):
            For each pair (pred_i, gt_j), we compute:
                d1 = ||P1 - G1|| + ||P2 - G2||
                d2 = ||P1 - G2|| + ||P2 - G1||
                d  = min(d1, d2) / 2
            A pair is a candidate if d <= tau.

        Then we greedily select the closest candidate pairs (smallest d first)
        such that each pred and each GT is matched at most once.

        Returns
        -------
        tp_vec : np.ndarray, shape [N_pred], int32, 1 if this prediction is TP
        fp_vec : np.ndarray, shape [N_pred], int32, 1 if this prediction is FP
        n_gt   : int, number of GT lines (for counting FN later)
        """
        pred = np.asarray(pred_lines, dtype=np.float32)
        gt = np.asarray(gt_lines, dtype=np.float32)

        Np = pred.shape[0]
        Ng = gt.shape[0]

        # Degenerate cases
        if Np == 0 and Ng == 0:
            return (
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                0,
            )

        if Np == 0:
            # No predictions -> all GT are FN, but we return empty tp/fp
            return (
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                Ng,
            )

        if Ng == 0:
            # No GT -> all preds are FP
            return (
                np.zeros(Np, dtype=np.int32),
                np.ones(Np, dtype=np.int32),
                0,
            )

        # Pred endpoints: [Np, 2]
        P1 = pred[:, 0, :]  # (y1, x1)
        P2 = pred[:, 1, :]  # (y2, x2)

        # GT endpoints: [Ng, 2]
        G1 = gt[:, 0, :]
        G2 = gt[:, 1, :]

        # Expand for broadcasting: [Np, Ng, 2]
        P1e = P1[:, None, :]
        P2e = P2[:, None, :]
        G1e = G1[None, :, :]
        G2e = G2[None, :, :]

        # Compute endpoint distances
        d1 = np.linalg.norm(P1e - G1e, axis=-1) + np.linalg.norm(P2e - G2e, axis=-1)
        d2 = np.linalg.norm(P1e - G2e, axis=-1) + np.linalg.norm(P2e - G1e, axis=-1)
        d = np.minimum(d1, d2) / 2.0  # average endpoint error, shape [Np, Ng]

        # Candidate pairs within threshold
        cand_indices = np.argwhere(d <= tau)  # [[pred_idx, gt_idx], ...]

        matched_pred = np.zeros(Np, dtype=bool)
        matched_gt = np.zeros(Ng, dtype=bool)

        if cand_indices.size > 0:
            # Sort candidates by distance (ascending)
            cand_dists = d[cand_indices[:, 0], cand_indices[:, 1]]
            order = np.argsort(cand_dists)
            cand_indices = cand_indices[order]

            # Greedy 1:1 matching
            for p_idx, g_idx in cand_indices:
                if not matched_pred[p_idx] and not matched_gt[g_idx]:
                    matched_pred[p_idx] = True
                    matched_gt[g_idx] = True

        tp_vec = matched_pred.astype(np.int32)
        fp_vec = (~matched_pred).astype(np.int32)
        n_gt = Ng

        return tp_vec, fp_vec, n_gt

    @staticmethod
    def _average_precision(recall, precision):
        """
        Compute AP given recall and precision curves.
        Uses the standard "area under the precision–recall curve"
        with precision envelope (like VOC-style, but continuous).

        Parameters
        ----------
        recall : np.ndarray, shape [N]
        precision : np.ndarray, shape [N]

        Returns
        -------
        float : AP in [0, 1]
        """
        if recall.size == 0:
            return 0.0

        # Append sentinel endpoints
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        # Make precision monotonically non-increasing
        for i in range(mpre.size - 2, -1, -1):
            if mpre[i] < mpre[i + 1]:
                mpre[i] = mpre[i + 1]

        # Sum over recall steps where it changes
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        return float(ap)
