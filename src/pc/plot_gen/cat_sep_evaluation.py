# eval_many_catsep_vs_gt_nocli.py
# Drop-in, no-argparse version of your evaluator.
# Call evaluate_catsep_vs_gt(pred_dir, gt_dir, ...) from your code or edit the __main__ block.

import json, csv, re
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np

# ---------- pairing & keys (unchanged) ----------

_CROP_RE = re.compile(r"(?:^|_)(crop)_(\d+)(?:_|$)", re.IGNORECASE)

def token_key(stem: str) -> str:
    """
    Order-agnostic key to pair pred<->gt for the SAME CROP.
    Example: 'image_1_crop_3_SOXR' and 'image_1_SOXR_crop_3' -> same key.
    """
    toks = [t for t in stem.split("_") if t]
    toks = [t.lower() for t in toks]
    return "_".join(sorted(toks))

def base_key(stem: str) -> str:
    """
    Order-agnostic key for grouping by BASE IMAGE (strip '_crop_#').
    Works even if 'crop_#' appears in a different position.
    """
    def _strip_one_crop_token(tokens: List[str]) -> List[str]:
        out = []
        skip_next = False
        for i, t in enumerate(tokens):
            if skip_next:
                skip_next = False
                continue
            if t.lower() == "crop" and i+1 < len(tokens) and tokens[i+1].isdigit():
                skip_next = True
                continue
            out.append(t)
        return out

    toks = [t for t in stem.split("_") if t]
    toks = [t.lower() for t in toks]
    toks = _strip_one_crop_token(toks)
    return "_".join(sorted(toks))

def collect_images(d: Path, exts=(".png",".jpg",".jpeg",".bmp")) -> Dict[str, Path]:
    m = {}
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            m[token_key(p.stem)] = p
    return m

# ---------- masks & metrics (unchanged) ----------

def load_mask(path: Path, white_thresh: int = 750) -> np.ndarray:
    """
    Build a binary mask from an image: non-white pixels are foreground (1).
    white_thresh is the sum(B,G,R) threshold; tweak if backgrounds aren't pure white.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    s = img.astype(np.uint16).sum(axis=2)
    return (s < white_thresh).astype(np.uint8)

def load_mask_hsv(path, s_thresh=20, v_thresh=230):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S = hsv[...,1]
    V = hsv[...,2]
    bg = (S < s_thresh) & (V > v_thresh)
    mask = (~bg).astype(np.uint8)  # 1 = foreground
    return mask

def metrics_from_masks(pred_mask: np.ndarray, gt_mask: np.ndarray):
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    tp = int(np.logical_and(pred_mask==1, gt_mask==1).sum())
    fp = int(np.logical_and(pred_mask==1, gt_mask==0).sum())
    fn = int(np.logical_and(pred_mask==0, gt_mask==1).sum())
    tn = int(np.logical_and(pred_mask==0, gt_mask==0).sum())
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = tp / (tp + fp + fn + eps)
    acc       = (tp + tn) / (tp + tn + fp + fn + eps)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "f1_dice": f1, "iou": iou, "accuracy": acc,
        "pred_fg_px": int(pred_mask.sum()),
        "gt_fg_px": int(gt_mask.sum())
    }

def combine_counts(rows: List[dict]) -> dict:
    """Micro-average: sum counts then recompute metrics."""
    s_tp = sum(r["tp"] for r in rows)
    s_fp = sum(r["fp"] for r in rows)
    s_fn = sum(r["fn"] for r in rows)
    s_tn = sum(r["tn"] for r in rows)
    eps = 1e-9
    precision = s_tp / (s_tp + s_fp + eps)
    recall    = s_tp / (s_tp + s_fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = s_tp / (s_tp + s_fp + s_fn + eps)
    acc       = (s_tp + s_tn) / (s_tp + s_tn + s_fp + s_fn + eps)
    return {
        "tp": s_tp, "fp": s_fp, "fn": s_fn, "tn": s_tn,
        "precision": precision, "recall": recall, "f1_dice": f1, "iou": iou, "accuracy": acc
    }

def mean_metrics(rows: List[dict]) -> dict:
    """Macro-average of the provided metric fields."""
    if not rows:
        return {"precision":0,"recall":0,"f1_dice":0,"iou":0,"accuracy":0}
    keys = ["precision","recall","f1_dice","iou","accuracy"]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}

# ---------- core evaluator (no argparse) ----------

def evaluate_catsep_vs_gt(
    pred_dir,
    gt_dir,
    white_thresh: int = 750,
    per_crop_csv: str = "per_crop_results.csv",
    per_base_csv: str = "per_base_results.csv",
    summary_json: str = "summary.json",
    verbose: bool = True
) -> dict:
    """
    Evaluate prediction images vs. GT images across many crops and bases.

    Args:
        pred_dir (str|Path): directory with prediction images.
        gt_dir   (str|Path): directory with GT images.
        white_thresh (int): sum(B,G,R) threshold to detect white background.
        per_crop_csv (str): path to write per-crop CSV.
        per_base_csv (str): path to write per-base CSV.
        summary_json (str): path to write summary JSON.
        verbose (bool): print progress summary.

    Returns:
        dict: summary payload (also written to summary_json).
    """
    pred_dir = Path(pred_dir)
    gt_dir   = Path(gt_dir)

    pred_map = collect_images(pred_dir)
    gt_map   = collect_images(gt_dir)

    pairs = []
    missing_pred, missing_gt = [], []

    # Pair using GT as driver (requires GT for evaluation)
    for k, gpath in gt_map.items():
        ppath = pred_map.get(k)
        if ppath is None:
            missing_pred.append(str(gpath))
        else:
            pairs.append((ppath, gpath, k))

    # Also track preds that had no GT
    for k, ppath in pred_map.items():
        if k not in gt_map:
            missing_gt.append(str(ppath))

    # ---- per-crop metrics ----
    per_crop_rows = []
    for ppath, gpath, key in pairs:
        try:
            pred_mask = load_mask(ppath, white_thresh)
            gt_mask   = load_mask(gpath, white_thresh)
            m = metrics_from_masks(pred_mask, gt_mask)
        except Exception as e:
            if verbose:
                print(f"[WARN] Skipping pair due to error: {ppath} <-> {gpath}: {e}")
            continue

        bkey = base_key(Path(ppath).stem)  # strip crop_#
        per_crop_rows.append({
            "base": bkey,
            "crop_key": key,
            "pred": str(ppath),
            "gt": str(gpath),
            **m
        })

    # write per-crop CSV
    if per_crop_rows:
        with open(per_crop_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_crop_rows[0].keys()))
            w.writeheader()
            w.writerows(per_crop_rows)

    # ---- per-base aggregation ----
    per_base_dict: Dict[str, List[dict]] = {}
    for r in per_crop_rows:
        per_base_dict.setdefault(r["base"], []).append(r)

    per_base_rows = []
    for b, rows in per_base_dict.items():
        micro = combine_counts(rows)      # micro per base
        macro = mean_metrics(rows)        # macro per base (optional)
        per_base_rows.append({
            "base": b,
            "num_crops": len(rows),
            **{f"micro_{k}": v for k,v in micro.items()},
            **{f"macro_{k}": v for k,v in macro.items()}
        })

    if per_base_rows:
        with open(per_base_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_base_rows[0].keys()))
            w.writeheader()
            w.writerows(per_base_rows)

    # ---- overall summary ----
    overall_micro = combine_counts(per_crop_rows)
    overall_macro_crops = mean_metrics(per_crop_rows)
    overall_macro_bases = mean_metrics(
        [{"precision": r["micro_precision"], "recall": r["micro_recall"],
          "f1_dice": r["micro_f1_dice"], "iou": r["micro_iou"], "accuracy": r["micro_accuracy"]}
         for r in per_base_rows]
    ) if per_base_rows else {"precision":0,"recall":0,"f1_dice":0,"iou":0,"accuracy":0}

    summary = {
        "counts": {
            "num_pairs": len(pairs),
            "num_per_crop_rows": len(per_crop_rows),
            "num_bases": len(per_base_rows),
            "num_gt_without_pred": len(missing_pred),
            "num_pred_without_gt": len(missing_gt),
        },
        "missing_gt_side": missing_gt[:100],
        "missing_pred_side": missing_pred[:100],
        "overall": {
            "micro": overall_micro,
            "macro_over_crops": overall_macro_crops,
            "macro_over_bases": overall_macro_bases
        }
    }

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(json.dumps(summary, indent=2))
        print(f"Per-crop CSV  -> {per_crop_csv}")
        print(f"Per-base CSV  -> {per_base_csv}")
        print(f"Summary JSON  -> {summary_json}")

    return summary

# Optional: edit these and run the file directly
# if __name__ == "__main__":
#     # <<< set these to your actual paths / filenames >>>
#     PRED_DIR = "/path/to/cat_sep_dir"
#     GT_DIR   = "/path/to/gt_dir"
#     evaluate_catsep_vs_gt(
#         pred_dir=PRED_DIR,
#         gt_dir=GT_DIR,
#         white_thresh=750,
#         per_crop_csv="per_crop_results.csv",
#         per_base_csv="per_base_results.csv",
#         summary_json="summary.json",
#         verbose=True
#     )
