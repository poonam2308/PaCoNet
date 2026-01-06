#!/usr/bin/env python3
"""
LLM Line Extraction Evaluation (Charts) — OpenAI only, no argparse, no color

- Iterates over images in IMAGE_DIR
- Reads GT from GT_JSON_PATH (single JSON) by matching entry['filename'] to image filename
- Calls OpenAI to predict line segments (x0,y0,x1,y1)
- Evaluates predictions vs GT using Hungarian matching + cost threshold
- Computes MAE for start, end, and all coords (NO endpoint swapping)
- Writes results to OUT_CSV

GT JSON format expected:
[
  {
    "filename": "image_1_crop_1.png",
    "lines": [
      [x0, y0, x1, y1],
      ...
    ]
  },
  ...
]

Model output accepted:
A) {"lines": [[x0,y0,x1,y1], ...]}
B) [[x0,y0,x1,y1], ...]
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


# =========================
# Config (edit these)
# =========================

# Your project_root logic (kept similar to your script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
project_root = Path(project_root)

IMAGE_DIR = project_root / "data/synthetic_plots/testing/images_100"
GT_JSON_PATH = project_root / "data/synthetic_plots/testing/test.json"

OUT_CSV = "results_openai_only.csv"

OPENAI_MODEL = "gpt-4.1-mini"  # or your preferred model

USE_OPENAI = True

MAX_IMAGES = 0     # 0 = no limit
MAX_SIDE = 0       # IMPORTANT: set to 0 to avoid resizing-based coord scale mismatch while evaluating
MAX_MATCH_COST = 400.0  # L2-based cost threshold; tune later when matching works

# Prompt (no color)
USER_PROMPT_BASE = """You are given a chart image with multiple straight line segments.

Task:
1) Detect all distinct straight line segments in the image.
2) Output endpoints in image pixel coordinates: [x0, y0, x1, y1].

Return ONLY valid JSON in exactly this format:
{
  "lines": [
    [x0, y0, x1, y1],
    ...
  ]
}

Rules:
- No markdown, no extra text.
- Coordinates are in image pixel coordinates with origin at top-left.
- Use numbers (ints or floats).
- Include every visible straight segment.
"""


# =========================
# API keys (you said you keep them at top)
# IMPORTANT: do not hardcode real keys in shared code.
# =========================

OPENAI_API_KEY="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"
#
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# -----------------------------
# Types
# -----------------------------
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


# -----------------------------
# IO helpers
# -----------------------------
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


# -----------------------------
# JSON parsing helpers
# -----------------------------
def safe_json_extract(text: str) -> Any:
    """
    Robustly parse JSON from model output.
    Supports top-level dict or list, possibly with extra text.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output.")

    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try dict block
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return json.loads(m.group(0))

    # Try list block
    m2 = re.search(r"\[[\s\S]*\]", text)
    if m2:
        return json.loads(m2.group(0))

    raise ValueError("No JSON object/array found in model output.")


def extract_lines_from_model_json(pred: Any) -> List[Line]:
    """
    Accepts:
      A) [[x0,y0,x1,y1], ...]
      B) {"lines": [[x0,y0,x1,y1], ...]}
    Returns: flat list[Line]
    """
    def parse_arr(arr: Any) -> List[Line]:
        out: List[Line] = []
        if not isinstance(arr, list):
            return out
        for item in arr:
            if isinstance(item, (list, tuple)) and len(item) == 4:
                try:
                    x0, y0, x1, y1 = map(float, item)
                    out.append((x0, y0, x1, y1))
                except Exception:
                    pass
        return out

    if isinstance(pred, list):
        return parse_arr(pred)
    if isinstance(pred, dict):
        return parse_arr(pred.get("lines", []))
    return []


def parse_gt_index(gt_payload: Any) -> Dict[str, List[Line]]:
    """
    GT payload expected:
      [
        {"filename": "...png", "lines": [[x0,y0,x1,y1], ...]},
        ...
      ]
    (Optionally wrapped in {"images":[...]} is also supported)
    """
    records = None
    if isinstance(gt_payload, list):
        records = gt_payload
    elif isinstance(gt_payload, dict) and isinstance(gt_payload.get("images"), list):
        records = gt_payload["images"]
    else:
        raise ValueError("Unsupported GT format. Expected a list or {'images': list}.")

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
            if isinstance(item, list) and len(item) == 4:
                try:
                    parsed.append((float(item[0]), float(item[1]), float(item[2]), float(item[3])))
                except Exception:
                    pass
        gt_index[fn] = parsed
    return gt_index


# -----------------------------
# Canonicalize endpoints (NO SWAPPING in evaluation)
# -----------------------------
def canonicalize_line(l: Line) -> Line:
    """Make (x0,y0) the leftmost endpoint; tie -> upper."""
    x0, y0, x1, y1 = l
    if (x1 < x0) or (x1 == x0 and y1 < y0):
        return (x1, y1, x0, y0)
    return (x0, y0, x1, y1)


# -----------------------------
# Matching + MAE (no swap)
# -----------------------------
def endpoint_cost_l2_noswap(a: Line, b: Line) -> float:
    """L2(start-start) + L2(end-end) AFTER canonicalization."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return float(np.hypot(ax0 - bx0, ay0 - by0) + np.hypot(ax1 - bx1, ay1 - by1))


def hungarian_pairs(cost: np.ndarray) -> List[Tuple[int, int]]:
    """Returns list of (gt_idx, pr_idx). Uses scipy if available, else greedy fallback."""
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    except Exception:
        pairs: List[Tuple[int, int]] = []
        used_j = set()
        for i in range(cost.shape[0]):
            best_j = None
            best_v = float("inf")
            for j in range(cost.shape[1]):
                if j in used_j:
                    continue
                v = float(cost[i, j])
                if v < best_v:
                    best_v = v
                    best_j = j
            if best_j is not None:
                used_j.add(best_j)
                pairs.append((i, best_j))
        return pairs


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

        # start abs (x0,y0) + end abs (x1,y1)
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

schema = {
    "type": "json_schema",
    "name": "line_segments",   # <-- REQUIRED (this is what your error complains about)
    "schema": {
        "type": "object",
        "properties": {
            "lines": {
                "type": "array",
                "items": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "number"},
                },
            }
        },
        "required": ["lines"],
        "additionalProperties": False,
    },
    "strict": True,
}

# -----------------------------
# OpenAI call (Responses API like your script)
# -----------------------------

from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI

class LinesOut(BaseModel):
    lines: List[List[float]] = Field(default_factory=list)


def call_openai_image_lines(img_b64: str, user_prompt: str) -> List[tuple[float,float,float,float]]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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
        text_format=LinesOut,   # <-- enforces schema
    )

    parsed = resp.output_parsed  # a LinesOut instance
    out = []
    for l in (parsed.lines or []):
        if isinstance(l, list) and len(l) == 4:
            try:
                out.append((float(l[0]), float(l[1]), float(l[2]), float(l[3])))
            except Exception:
                pass
    return out


# -----------------------------
# Main (style similar to your script, NO argparse)
# -----------------------------
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

    rows: List[Dict[str, Any]] = []
    tot_oai = MAEStats()

    for k, img_path in enumerate(img_paths, start=1):
        fn = img_path.name
        gt_lines = gt_index.get(fn)
        if gt_lines is None:
            print(f"[WARN] No GT found for {fn}, skipping.", file=sys.stderr)
            continue

        img_b64 = image_to_base64_png(img_path, max_side=MAX_SIDE)

        # ---- OpenAI ----
        oai_lines: List[Line] = []
        oai_stats = MAEStats(unmatched_gt=len(gt_lines), unmatched_pred=0)

        if USE_OPENAI:
            try:
                oai_lines = call_openai_image_lines(img_b64, USER_PROMPT_BASE)
                oai_stats = compute_mae_stats(gt_lines, oai_lines)
            except Exception as e:
                print(f"[WARN] OpenAI failed on {fn}: {e}", file=sys.stderr)

        # accumulate totals
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

        time.sleep(0.2)

    # finalize totals
    tot_oai.finalize()

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
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

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
            }
        )

    print(f"\nSaved CSV: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
