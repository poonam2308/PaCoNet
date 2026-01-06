#!/usr/bin/env python3
"""
chatgpt_all_with_sap.py

OpenAI(ChatGPT) only evaluation:
- MAE(start/end/all) after Hungarian matching (NO endpoint swapping; canonicalize endpoints)
- sAP (line segment average precision) using your sap_metric.py (dataset-level sAP5/sAP10/sAP15)

GT JSON expected (list):
[
  {"filename": "...png", "lines": [[x0,y0,x1,y1], ...]},
  ...
]

OpenAI output enforced via responses.parse (Pydantic schema):
{"lines": [[x0,y0,x1,y1], ...]}
"""

import base64
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

# Hungarian assignment
from scipy.optimize import linear_sum_assignment

# OpenAI + structured parsing
from openai import OpenAI
from pydantic import BaseModel, Field

from llms.chatgpt_all import GT_JSON_PATH
# Your sAP metric (must be importable; keep sap_metric.py in same folder or PYTHONPATH)
from sap_metric import LineSegmentSAPMetric


# =========================
# Config (edit these)
# =========================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
project_root = Path(project_root)

# IMAGE_DIR = project_root / "data/synthetic_plots/testing/images_100"
# GT_JSON_PATH = project_root / "data/synthetic_plots/testing/test.json"
# OUT_CSV = "results_openai_only_with_sap1.csv"

IMAGE_DIR = project_root / "data/synthetic_plots/multi_cat/testing/m_crops/images_224"
GT_JSON_PATH = project_root / "data/synthetic_plots/multi_cat/testing/m_crops/test.json"

OUT_CSV = project_root /" outputs/llms/results_openai_only_with_sap_test.csv"

OPENAI_MODEL = "gpt-4.1-mini"  # change if you want

USE_OPENAI = True
MAX_IMAGES = 0        # 0 = no limit
MAX_SIDE = 0          # 0 = do NOT resize (recommended to match GT pixel scale)
MAX_MATCH_COST = 400.0  # threshold on cost for match acceptance

# Your API key is "mentioned at the top" — keep it here if you want, but env var is safer:
OPENAI_API_KEY="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"
#
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


USER_PROMPT_BASE = """Extract all visible straight line segments (plotted data lines) in the image.

Return ONLY valid JSON in exactly this format:
{
  "lines": [
    [x0, y0, x1, y1],
    ...
  ]
}

Rules:
- No markdown, no extra text.
- Coordinates are IMAGE PIXELS with origin at top-left.
- Ignore axes, tick marks, grid lines, legend boxes, and text.
- Use numbers (ints or floats).
"""


# =========================
# Types / data
# =========================
Line = Tuple[float, float, float, float]


@dataclass
class MAEStats:
    matched: int = 0
    unmatched_gt: int = 0
    unmatched_pred: int = 0

    mae_start: float = 0.0
    mae_end: float = 0.0
    mae_all: float = 0.0

    _sum_abs_start: float = 0.0
    _sum_abs_end: float = 0.0
    _sum_abs_all: float = 0.0

    def finalize(self) -> None:
        if self.matched > 0:
            self.mae_start = self._sum_abs_start / (self.matched * 2.0)
            self.mae_end = self._sum_abs_end / (self.matched * 2.0)
            self.mae_all = self._sum_abs_all / (self.matched * 4.0)
        else:
            self.mae_start = 0.0
            self.mae_end = 0.0
            self.mae_all = 0.0


# =========================
# Helpers
# =========================
def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def image_to_base64_png(path: Path, max_side: int) -> str:
    """Encode to PNG base64. If max_side==0, no resizing."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max_side and max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)

    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_gt_index(gt_payload: Any) -> Dict[str, List[Line]]:
    """
    GT payload expected:
      [
        {"filename": "...png", "lines": [[x0,y0,x1,y1], ...]},
        ...
      ]
    Optionally wrapped:
      {"images":[...]}
    """
    if isinstance(gt_payload, list):
        records = gt_payload
    elif isinstance(gt_payload, dict) and isinstance(gt_payload.get("images"), list):
        records = gt_payload["images"]
    else:
        raise ValueError("Unsupported GT format. Expected list or {'images': list}.")

    gt_index: Dict[str, List[Line]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        fn = rec.get("filename")
        arr = rec.get("lines")
        if not isinstance(fn, str) or not isinstance(arr, list):
            continue
        parsed: List[Line] = []
        for item in arr:
            if isinstance(item, (list, tuple)) and len(item) == 4:
                try:
                    parsed.append((float(item[0]), float(item[1]), float(item[2]), float(item[3])))
                except Exception:
                    pass
        gt_index[fn] = parsed
    return gt_index


def canonicalize_line(l: Line) -> Line:
    """Make (x0,y0) the leftmost endpoint; tie -> upper."""
    x0, y0, x1, y1 = l
    if (x1 < x0) or (x1 == x0 and y1 < y0):
        return (x1, y1, x0, y0)
    return (x0, y0, x1, y1)


# =========================
# MAE matching (no swap)
# =========================
def endpoint_cost_l2_noswap(a: Line, b: Line) -> float:
    """L2(start-start) + L2(end-end) after canonicalization."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return float(np.hypot(ax0 - bx0, ay0 - by0) + np.hypot(ax1 - bx1, ay1 - by1))


def hungarian_pairs(cost: np.ndarray) -> List[Tuple[int, int]]:
    r, c = linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


def compute_mae_stats(gt_lines_in: List[Line], pr_lines_in: List[Line]) -> MAEStats:
    stats = MAEStats()

    gt_lines = [canonicalize_line(l) for l in gt_lines_in]
    pr_lines = [canonicalize_line(l) for l in pr_lines_in]

    n_gt = len(gt_lines)
    n_pr = len(pr_lines)

    if n_gt == 0 and n_pr == 0:
        stats.finalize()
        return stats
    if n_gt == 0:
        stats.unmatched_pred = n_pr
        stats.finalize()
        return stats
    if n_pr == 0:
        stats.unmatched_gt = n_gt
        stats.finalize()
        return stats

    cost = np.zeros((n_gt, n_pr), dtype=float)
    for i in range(n_gt):
        for j in range(n_pr):
            cost[i, j] = endpoint_cost_l2_noswap(gt_lines[i], pr_lines[j])

    pairs = hungarian_pairs(cost)

    matched_gt = set()
    matched_pr = set()

    for i, j in pairs:
        c = float(cost[i, j])
        if c > MAX_MATCH_COST:
            continue

        matched_gt.add(i)
        matched_pr.add(j)

        gx0, gy0, gx1, gy1 = gt_lines[i]
        px0, py0, px1, py1 = pr_lines[j]

        s_abs = abs(gx0 - px0) + abs(gy0 - py0)
        e_abs = abs(gx1 - px1) + abs(gy1 - py1)

        stats._sum_abs_start += s_abs
        stats._sum_abs_end += e_abs
        stats._sum_abs_all += (s_abs + e_abs)
        stats.matched += 1

    stats.unmatched_gt = n_gt - len(matched_gt)
    stats.unmatched_pred = n_pr - len(matched_pr)

    stats.finalize()
    return stats


# =========================
# sAP helpers (convert xyxy -> [N,2,2] in (y,x))
# =========================
def xyxy_list_to_yx_tensor(lines_xyxy: List[Line]) -> np.ndarray:
    """
    Convert [(x0,y0,x1,y1), ...] -> np.ndarray [N,2,2] with (y,x) endpoints.
    """
    if not lines_xyxy:
        return np.zeros((0, 2, 2), dtype=np.float32)
    arr = np.asarray(lines_xyxy, dtype=np.float32)  # [N,4]
    yx = np.stack(
        [
            np.stack([arr[:, 1], arr[:, 0]], axis=-1),  # (y0, x0)
            np.stack([arr[:, 3], arr[:, 2]], axis=-1),  # (y1, x1)
        ],
        axis=1,
    )
    return yx.astype(np.float32)


def scores_from_length(lines_xyxy: List[Line]) -> np.ndarray:
    """Proxy confidence scores for ChatGPT preds: segment length."""
    if not lines_xyxy:
        return np.zeros((0,), dtype=np.float32)
    arr = np.asarray(lines_xyxy, dtype=np.float32)
    dx = arr[:, 2] - arr[:, 0]
    dy = arr[:, 3] - arr[:, 1]
    return np.sqrt(dx * dx + dy * dy).astype(np.float32)


# =========================
# OpenAI structured output
# =========================
class LinesOut(BaseModel):
    lines: List[List[float]] = Field(default_factory=list)


def call_openai_image_lines(img_b64: str, user_prompt: str) -> List[Line]:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set (env or top of script).")

    client = OpenAI(api_key=api_key)

    resp = client.responses.parse(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"},
                ],
            }
        ],
        text_format=LinesOut,
    )

    parsed: LinesOut = resp.output_parsed
    out: List[Line] = []
    for l in (parsed.lines or []):
        if isinstance(l, list) and len(l) == 4:
            try:
                out.append((float(l[0]), float(l[1]), float(l[2]), float(l[3])))
            except Exception:
                pass
    return out


# =========================
# Main (no argparse, like your style)
# =========================
def main() -> None:
    images_dir = Path(IMAGE_DIR)
    gt_path = Path(GT_JSON_PATH)
    out_csv = Path(OUT_CSV)

    if not images_dir.exists():
        raise RuntimeError(f"IMAGE_DIR not found: {images_dir}")
    if not gt_path.exists():
        raise RuntimeError(f"GT_JSON_PATH not found: {gt_path}")

    img_paths = sorted([p for p in images_dir.iterdir() if p.is_file() and is_image_file(p)])
    if MAX_IMAGES and MAX_IMAGES > 0:
        img_paths = img_paths[:MAX_IMAGES]
    if not img_paths:
        raise RuntimeError(f"No images found in: {images_dir}")

    gt_index = parse_gt_index(load_json(gt_path))

    # sAP metric (dataset-level)
    sap_metric = LineSegmentSAPMetric(thresholds=(5.0, 10.0, 15.0))

    # Totals for MAE
    tot_oai = MAEStats()

    # CSV rows
    rows: List[Dict[str, Any]] = []

    for k, img_path in enumerate(img_paths, start=1):
        fn = img_path.name
        gt_lines = gt_index.get(fn)
        if gt_lines is None:
            print(f"[WARN] No GT found for {fn}, skipping.", file=sys.stderr)
            continue

        img_b64 = image_to_base64_png(img_path, max_side=MAX_SIDE)

        # ---- OpenAI preds ----
        oai_lines: List[Line] = []
        oai_stats = MAEStats(unmatched_gt=len(gt_lines), unmatched_pred=0)

        if USE_OPENAI:
            try:
                oai_lines = call_openai_image_lines(img_b64, USER_PROMPT_BASE)
                if oai_lines:
                    xs = [v for l in oai_lines for v in (l[0], l[2])]
                    ys = [v for l in oai_lines for v in (l[1], l[3])]
                    print(f"[DBG] {fn} pred x:[{min(xs):.2f},{max(xs):.2f}] y:[{min(ys):.2f},{max(ys):.2f}]")

                oai_stats = compute_mae_stats(gt_lines, oai_lines)

                # ---- sAP add_image ----
                pred_lines_yx = xyxy_list_to_yx_tensor(oai_lines)
                gt_lines_yx = xyxy_list_to_yx_tensor(gt_lines)
                pred_scores = scores_from_length(oai_lines)
                sap_metric.add_image(pred_lines_yx, pred_scores, gt_lines_yx)

            except Exception as e:
                print(f"[WARN] OpenAI failed on {fn}: {e}", file=sys.stderr)

        # accumulate totals (MAE)
        tot_oai.matched += oai_stats.matched
        tot_oai.unmatched_gt += oai_stats.unmatched_gt
        tot_oai.unmatched_pred += oai_stats.unmatched_pred
        tot_oai._sum_abs_start += oai_stats._sum_abs_start
        tot_oai._sum_abs_end += oai_stats._sum_abs_end
        tot_oai._sum_abs_all += oai_stats._sum_abs_all

        rows.append(
            {
                "image": fn,
                "gt_lines": len(gt_lines),
                "oai_pred_lines": len(oai_lines),
                "oai_matched": oai_stats.matched,
                "oai_unmatched_gt": oai_stats.unmatched_gt,
                "oai_unmatched_pred": oai_stats.unmatched_pred,
                "oai_mae_start": oai_stats.mae_start,
                "oai_mae_end": oai_stats.mae_end,
                "oai_mae_all": oai_stats.mae_all,
            }
        )

        print(
            f"[{k}/{len(img_paths)}] {fn} | GT={len(gt_lines)} | "
            f"OAI: pred={len(oai_lines)} matched={oai_stats.matched} "
            f"MAE(start/end/all)={oai_stats.mae_start:.3f}/{oai_stats.mae_end:.3f}/{oai_stats.mae_all:.3f}"
        )

        time.sleep(0.15)

    # finalize MAE totals
    tot_oai.finalize()

    # compute sAP totals
    sap = sap_metric.compute_sap()  # dict keyed by thresholds

    print("\n=============== TOTAL (MAE) ===============")
    print(
        f"matched={tot_oai.matched} | "
        f"MAE(start/end/all)={tot_oai.mae_start:.3f}/{tot_oai.mae_end:.3f}/{tot_oai.mae_all:.3f}"
    )

    print("\n=============== sAP (dataset) ===============")
    print(f"sAP5  = {sap.get(5.0, 0.0):.6f}")
    print(f"sAP10 = {sap.get(10.0, 0.0):.6f}")
    print(f"sAP15 = {sap.get(15.0, 0.0):.6f}")
    print(f"sAP5%  = {100.0 * sap.get(5.0, 0.0):.3f}%")
    print(f"sAP10% = {100.0 * sap.get(10.0, 0.0):.3f}%")
    print(f"sAP15% = {100.0 * sap.get(15.0, 0.0):.3f}%")

    # write CSV
    fieldnames = [
        "image",
        "gt_lines",
        "oai_pred_lines",
        "oai_matched",
        "oai_unmatched_gt",
        "oai_unmatched_pred",
        "oai_mae_start",
        "oai_mae_end",
        "oai_mae_all",
        # sAP written in TOTAL row only:
        "sap5",
        "sap10",
        "sap15",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for r in rows:
            r2 = dict(r)
            r2["sap5"] = ""
            r2["sap10"] = ""
            r2["sap15"] = ""
            w.writerow(r2)

        w.writerow(
            {
                "image": "TOTAL",
                "gt_lines": "",
                "oai_pred_lines": "",
                "oai_matched": tot_oai.matched,
                "oai_unmatched_gt": tot_oai.unmatched_gt,
                "oai_unmatched_pred": tot_oai.unmatched_pred,
                "oai_mae_start": tot_oai.mae_start,
                "oai_mae_end": tot_oai.mae_end,
                "oai_mae_all": tot_oai.mae_all,
                "sap5": sap.get(5.0, 0.0),
                "sap10": sap.get(10.0, 0.0),
                "sap15": sap.get(15.0, 0.0),
            }
        )

    print(f"\nSaved CSV: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
