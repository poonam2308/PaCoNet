#!/usr/bin/env python3
"""
LLM Line Extraction Evaluation (Charts)

What it does
- Iterates over images in IMAGE_DIR
- Reads GT from one JSON file (GT_JSON_PATH) by matching entry['filename'] to image filename
- Calls OpenAI and Gemini to predict line segments (x0,y0,x1,y1)
- Evaluates predictions vs GT using Hungarian matching + cost threshold
- Writes results to OUT_CSV and saves visualization overlays in VIZ_DIR

GT JSON format expected (single file):
[
  {
    "filename": "image_1_crop_1.png",
    "lines": [
      [0.0, 19.3, 224.0, 123.89],
      ...
    ]
  },
  ...
]
"""


import argparse
import base64
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))  # 2 levels up from this file
sys.path.insert(0, project_root)

project_root = Path(project_root)
# -----------------------------
# Config
# -----------------------------


IMAGE_DIR = project_root / "data/synthetic_plots/testing/images_100"  # directory containing images
GT_JSON_PATH = project_root / "data/synthetic_plots/testing/test.json"  # single JSON containing per-image GT

OUT_CSV = "results.csv"  # where to save results
VIZ_DIR = Path("./viz")

OPENAI_MODEL = "gpt-4.1-mini"
GEMINI_MODEL = "gemini-1.5-pro-latest"

# If you want to disable a provider, set USE_OPENAI / USE_GEMINI = False
USE_OPENAI = True
USE_GEMINI = True

# Limits
MAX_IMAGES = 0  # 0 = no limit
MAX_SIDE = 1024  # resize images so max(width,height) <= this when encoding for LLM

# Matching thresholds (tune as needed)
MAX_MATCH_COST = 60.0  # max allowed cost for a match (lower = stricter)

# Visualization settings
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Prompts

USER_PROMPT_BASE = """You are given a chart image with multiple colored straight line segments.

Task:
1) Detect all distinct straight line segments in the image.
2) Determine the color of each line segment and group the segments by color.

Color categories:
- Use one of these basic color names when possible: red, blue, green, yellow, orange, purple, pink, brown, black, gray, white, cyan, magenta.
- If a line is a shade (e.g., light blue), still output the closest basic name (e.g., blue).

Output endpoints in image pixel coordinates: [x0, y0, x1, y1].
Return ONLY valid JSON in exactly this format:
{
  "lines": [
    [x0, y0, x1, y1],
    ...
  ]
}

Rules:
- No markdown, no extra text.
- Coordinates are in image pixel coordinates.
- Use numbers (ints or floats).
- Include every visible straight segment.
"""


# =========================

OPENAI_API_KEY="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"
GEMINI_API_KEY="AIzaSyDnmum8L8pHicm-8OqNOFzuTeG7fxI1yCM"


# Put keys into env so the rest of the code can read them
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY



# -----------------------------
# Types
# -----------------------------
Line = Tuple[float, float, float, float]


@dataclass
class MAEStats:
    matched: int = 0
    unmatched_gt: int = 0
    unmatched_pred: int = 0

    mae_start: float = 0.0  # mean abs error over x0,y0 across matched pairs
    mae_end: float = 0.0    # mean abs error over x1,y1 across matched pairs
    mae_all: float = 0.0    # mean abs error over x0,y0,x1,y1 across matched pairs

    # Internals for accumulation
    _sum_abs_start: float = 0.0
    _sum_abs_end: float = 0.0
    _sum_abs_all: float = 0.0

    def finalize(self) -> None:
        if self.matched > 0:
            # each start has 2 coords, end has 2 coords, all has 4 coords
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


def extract_lines_by_color(model_json: Dict[str, Any]) -> Dict[str, List[Line]]:
    out: Dict[str, List[Line]] = {}

    lines_root = (model_json or {}).get("lines", {})
    crop_all = (lines_root or {}).get("crop_all", {})

    # crop_all should be { "red": [[...]], "blue": [[...]], ... }
    if isinstance(crop_all, dict):
        for color, arr in crop_all.items():
            if not isinstance(arr, list):
                continue
            parsed: List[Line] = []
            for item in arr:
                if (
                    isinstance(item, (list, tuple)) and len(item) == 4
                    and all(isinstance(v, (int, float)) for v in item)
                ):
                    x0, y0, x1, y1 = map(float, item)
                    parsed.append((x0, y0, x1, y1))
            if parsed:
                out[str(color).lower().strip()] = parsed

    return out
def extract_lines_from_model_json(pred: Any) -> List[Line]:
    """
    Accepts any of:
      A) [[x0,y0,x1,y1], ...]
      B) {"lines": [[x0,y0,x1,y1], ...]}
      C) {"lines": {"crop_all": {"unknown": [[...]]}}}
      D) {"lines": {"crop_all": {"red":[...], "blue":[...], ...}}}
    Returns a flat list[Line].
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

    # A) top-level list
    if isinstance(pred, list):
        return parse_arr(pred)

    if not isinstance(pred, dict):
        return []

    lines = pred.get("lines")

    # B) {"lines": [...]}
    if isinstance(lines, list):
        return parse_arr(lines)

    # C/D) {"lines": {"crop_all": {...}}}
    if isinstance(lines, dict):
        crop_all = lines.get("crop_all")
        if isinstance(crop_all, dict):
            out: List[Line] = []
            for _, arr in crop_all.items():
                out.extend(parse_arr(arr))
            return out

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
            if (
                isinstance(item, list)
                and len(item) == 4
                and all(isinstance(v, (int, float)) for v in item)
            ):
                parsed.append((float(item[0]), float(item[1]), float(item[2]), float(item[3])))
        gt_index[fn] = parsed
    return gt_index


# -----------------------------
# Matching + MAE
# -----------------------------
def safe_json_extract(text: str) -> Any:   # <- was Dict[str, Any]
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        # ALSO try list fallback
        m2 = re.search(r"\[[\s\S]*\]", text)
        if not m2:
            raise ValueError("No JSON found in model output.")
        return json.loads(m2.group(0))
    return json.loads(m.group(0))


def canonicalize_line(l: Line) -> Line:
    """Ensure (x0,y0) is the 'start' (leftmost; tie -> upper)."""
    x0, y0, x1, y1 = l
    if (x1 < x0) or (x1 == x0 and y1 < y0):
        return (x1, y1, x0, y0)
    return (x0, y0, x1, y1)


def endpoint_cost_l2_noswap(a: Line, b: Line) -> float:
    """Cost after canonicalization: L2(start-start) + L2(end-end)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return float(np.hypot(ax0 - bx0, ay0 - by0) + np.hypot(ax1 - bx1, ay1 - by1))


def abs_errors_noswap(gt: Line, pr: Line) -> Tuple[float, float, float]:
    """Return sums of abs errors for start/end/all (no swapping)."""
    gx0, gy0, gx1, gy1 = gt
    px0, py0, px1, py1 = pr

    start = abs(gx0 - px0) + abs(gy0 - py0)
    end   = abs(gx1 - px1) + abs(gy1 - py1)
    return start, end, start + end


def compute_mae_stats(gt_lines: List[Line], pr_lines: List[Line]) -> MAEStats:
    stats = MAEStats()

    # canonicalize first
    gt_lines = [canonicalize_line(l) for l in gt_lines]
    pr_lines = [canonicalize_line(l) for l in pr_lines]

    n_gt, n_pr = len(gt_lines), len(pr_lines)

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

        s_abs, e_abs, all_abs = abs_errors_noswap(gt_lines[i], pr_lines[j])
        stats._sum_abs_start += s_abs
        stats._sum_abs_end += e_abs
        stats._sum_abs_all += all_abs
        stats.matched += 1

    stats.unmatched_gt = n_gt - len(matched_gt)
    stats.unmatched_pred = n_pr - len(matched_pr)
    stats.finalize()
    return stats

def endpoint_cost_l2(a: Line, b: Line) -> float:
    """
    Matching cost used for Hungarian assignment:
    min( L2(start-start)+L2(end-end), L2(start-end)+L2(end-start) )
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    d_same = np.hypot(ax0 - bx0, ay0 - by0) + np.hypot(ax1 - bx1, ay1 - by1)
    d_swap = np.hypot(ax0 - bx1, ay0 - by1) + np.hypot(ax1 - bx0, ay1 - by0)
    return float(min(d_same, d_swap))


def best_orientation_abs_errors(gt: Line, pr: Line) -> Tuple[float, float, float]:
    """
    For a matched pair, compute abs errors for:
      - start (x0,y0)
      - end   (x1,y1)
      - all   (x0,y0,x1,y1)
    choosing the orientation (same vs swapped) that minimizes total abs error.
    Returns sums of abs errors (not averaged).
    """
    gx0, gy0, gx1, gy1 = gt
    px0, py0, px1, py1 = pr

    # same orientation
    same_start = abs(gx0 - px0) + abs(gy0 - py0)
    same_end = abs(gx1 - px1) + abs(gy1 - py1)
    same_all = same_start + same_end

    # swapped orientation
    swap_start = abs(gx0 - px1) + abs(gy0 - py1)
    swap_end = abs(gx1 - px0) + abs(gy1 - py0)
    swap_all = swap_start + swap_end

    if swap_all < same_all:
        return swap_start, swap_end, swap_all
    return same_start, same_end, same_all


def hungarian_pairs(cost: np.ndarray) -> List[Tuple[int, int]]:
    """
    Returns list of (gt_idx, pr_idx). Uses scipy if available, else greedy fallback.
    """
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    except Exception:
        # Greedy fallback
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


# -----------------------------
# OpenAI + Gemini calls
# -----------------------------
def call_openai_image_json(img_b64: str, user_prompt: str) -> Dict[str, Any]:
    """
    Requires env var OPENAI_API_KEY.
    """
    from openai import OpenAI  # type: ignore

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    client = OpenAI(api_key=api_key)

    resp = client.responses.create(
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
    )
    return safe_json_extract(resp.output_text)


def call_gemini_image_json(img_b64: str, user_prompt: str) -> Dict[str, Any]:
    """
    Uses Google GenAI SDK (pip install google-genai)
    Requires GEMINI_API_KEY or GOOGLE_API_KEY.
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in environment.")

    client = genai.Client(api_key=api_key)

    image_bytes = base64.b64decode(img_b64.encode("utf-8"))

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        # Pass prompt as a string (no Part.from_text)
        contents=[
            user_prompt,
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    text = getattr(resp, "text", "") or ""
    return safe_json_extract(text)




# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, default=str(IMAGE_DIR))
    parser.add_argument("--gt", type=str, default=str(GT_JSON_PATH))
    parser.add_argument("--out", type=str, default=OUT_CSV)
    args = parser.parse_args()

    images_dir = Path(args.images)
    gt_path = Path(args.gt)
    out_csv = Path(args.out)

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

    # Totals (weighted by matched pairs/coords, not by images)
    tot_oai = MAEStats()
    tot_gem = MAEStats()

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
                pred = call_openai_image_json(img_b64, USER_PROMPT_BASE)
                oai_lines = extract_lines_from_model_json(pred)
                oai_by_color = {}
                if isinstance(pred, dict):
                    oai_by_color = extract_lines_by_color(pred)
                oai_stats = compute_mae_stats(gt_lines, oai_lines)
            except Exception as e:
                print(f"[WARN] OpenAI failed on {fn}: {e}", file=sys.stderr)

        # ---- Gemini ----
        gem_lines: List[Line] = []
        gem_stats = MAEStats(unmatched_gt=len(gt_lines), unmatched_pred=0)
        if USE_GEMINI:
            try:
                pred = call_gemini_image_json(img_b64, USER_PROMPT_BASE)
                gem_lines = extract_lines_from_model_json(pred)
                gem_stats = compute_mae_stats(gt_lines, gem_lines)
            except Exception as e:
                print(f"[WARN] Gemini failed on {fn}: {e}", file=sys.stderr)

        # accumulate totals (sum abs errors + counts)
        # OpenAI totals
        tot_oai.matched += oai_stats.matched
        tot_oai.unmatched_gt += oai_stats.unmatched_gt
        tot_oai.unmatched_pred += oai_stats.unmatched_pred
        tot_oai._sum_abs_start += oai_stats._sum_abs_start
        tot_oai._sum_abs_end += oai_stats._sum_abs_end
        tot_oai._sum_abs_all += oai_stats._sum_abs_all

        # Gemini totals
        tot_gem.matched += gem_stats.matched
        tot_gem.unmatched_gt += gem_stats.unmatched_gt
        tot_gem.unmatched_pred += gem_stats.unmatched_pred
        tot_gem._sum_abs_start += gem_stats._sum_abs_start
        tot_gem._sum_abs_end += gem_stats._sum_abs_end
        tot_gem._sum_abs_all += gem_stats._sum_abs_all

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
                "gem_pred_lines": len(gem_lines),
                "gem_matched": gem_stats.matched,
                "gem_unmatched_gt": gem_stats.unmatched_gt,
                "gem_unmatched_pred": gem_stats.unmatched_pred,
                "gem_mae_start": gem_stats.mae_start,
                "gem_mae_end": gem_stats.mae_end,
                "gem_mae_all": gem_stats.mae_all,
            }
        )

        print(
            f"[{k}/{len(img_paths)}] {fn} | GT={len(gt_lines)} | "
            f"OAI: pred={len(oai_lines)} matched={oai_stats.matched} "
            f"MAE(start/end/all)={oai_stats.mae_start:.3f}/{oai_stats.mae_end:.3f}/{oai_stats.mae_all:.3f} | "
            f"GEM: pred={len(gem_lines)} matched={gem_stats.matched} "
            f"MAE(start/end/all)={gem_stats.mae_start:.3f}/{gem_stats.mae_end:.3f}/{gem_stats.mae_all:.3f}"
        )

        time.sleep(0.2)

    # finalize totals
    tot_oai.finalize()
    tot_gem.finalize()

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
        "gem_pred_lines",
        "gem_matched",
        "gem_unmatched_gt",
        "gem_unmatched_pred",
        "gem_mae_start",
        "gem_mae_end",
        "gem_mae_all",
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
                "gem_pred_lines": "",
                "gem_matched": tot_gem.matched,
                "gem_unmatched_gt": tot_gem.unmatched_gt,
                "gem_unmatched_pred": tot_gem.unmatched_pred,
                "gem_mae_start": tot_gem.mae_start,
                "gem_mae_end": tot_gem.mae_end,
                "gem_mae_all": tot_gem.mae_all,
            }
        )

    print(f"\nSaved CSV: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
