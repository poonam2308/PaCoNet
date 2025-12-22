#!/usr/bin/env python3
"""
Evaluate ChatGPT (OpenAI vision) vs Gemini (Google vision) for extracting
line start/end points from chart images, then compute MAE vs GT.

SETUP:
  pip install pillow openai google-generativeai
  # Optional for optimal matching:
  pip install scipy numpy

KEYS (env vars):
  export OPENAI_API_KEY="..."
  export GEMINI_API_KEY="..."

DIRECTORY EXPECTATIONS:
  IMAGES_DIR contains image files (png/jpg/etc)
  GT_JSON_DIR contains matching GT json files named: <image_stem>.json

OUTPUT:
  Writes CSV results to OUT_CSV
"""

import base64
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))  # 2 levels up from this file
sys.path.insert(0, project_root)

project_root = Path(project_root)
# =========================
# CONFIG (EDIT THESE)
# =========================
IMAGES_DIR = project_root / "data/synthetic_plots/testing/images_100"    # directory containing your 100 chart images
GT_JSON_DIR = project_root / "data/synthetic_plots/testing/images_100"     # directory containing GT json annotations
OUT_CSV = "results.csv"        # where to save results

OPENAI_MODEL = "gpt-4.1-mini"    # or "gpt-4.1" etc.
GEMINI_MODEL = "gemini-1.5-pro"  # or "gemini-1.5-flash"

TEMPERATURE = 0.0
MAX_SIDE = 1600                  # downscale max side before sending to API
MAX_IMAGES = 0                   # 0 = all images, else limit (e.g. 10)

OPENAI_API_KEY="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"
GEMINI_API_KEY="AIzaSyDnmum8L8pHicm-8OqNOFzuTeG7fxI1yCM"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
# =========================


# -----------------------------
# Optional Hungarian assignment
# -----------------------------
def hungarian_assign(cost: List[List[float]]) -> List[Tuple[int, int]]:
    """
    Returns list of (row_i, col_j) assignments minimizing total cost.
    Prefers scipy if available; otherwise greedy fallback.
    """
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
        import numpy as np  # type: ignore

        c = np.array(cost, dtype=float)
        r, cidx = linear_sum_assignment(c)
        return list(zip(r.tolist(), cidx.tolist()))
    except Exception:
        # Greedy fallback
        assigned_cols = set()
        pairs = []
        for i in range(len(cost)):
            best_j = None
            best_v = float("inf")
            for j, v in enumerate(cost[i]):
                if j in assigned_cols:
                    continue
                if v < best_v:
                    best_v = v
                    best_j = j
            if best_j is not None:
                assigned_cols.add(best_j)
                pairs.append((i, best_j))
        return pairs


# -----------------------------
# Data structures
# -----------------------------
Line = Tuple[float, float, float, float]  # (x0,y0,x1,y1)


@dataclass
class EvalStats:
    n_matched: int = 0
    mae_xy: float = 0.0
    mae_y: float = 0.0
    unmatched_pred: int = 0
    unmatched_gt: int = 0


# -----------------------------
# Utilities
# -----------------------------
def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def image_to_base64_png(path: Path, max_side: int = 1600) -> str:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def safe_json_extract(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))


def normalize_pred_to_lines(pred: Dict[str, Any]) -> Dict[str, Dict[str, List[Line]]]:
    if "lines" not in pred or not isinstance(pred["lines"], dict):
        raise ValueError("Prediction JSON must contain top-level key 'lines' as an object.")
    out: Dict[str, Dict[str, List[Line]]] = {}
    for crop, cats in pred["lines"].items():
        if not isinstance(cats, dict):
            continue
        out[crop] = {}
        for cat, arr in cats.items():
            if not isinstance(arr, list):
                continue
            lines: List[Line] = []
            for item in arr:
                if (
                    isinstance(item, list)
                    and len(item) == 4
                    and all(isinstance(v, (int, float)) for v in item)
                ):
                    lines.append((float(item[0]), float(item[1]), float(item[2]), float(item[3])))
            out[crop][str(cat)] = lines
    return out


def gt_lines_only(gt: Dict[str, Any]) -> Dict[str, Dict[str, List[Line]]]:
    if "lines" not in gt or not isinstance(gt["lines"], dict):
        raise ValueError("GT JSON missing 'lines' object.")
    out: Dict[str, Dict[str, List[Line]]] = {}
    for crop, cats in gt["lines"].items():
        if not isinstance(cats, dict):
            continue
        out[crop] = {}
        for cat, arr in cats.items():
            if not isinstance(arr, list):
                continue
            lines: List[Line] = []
            for item in arr:
                if (
                    isinstance(item, list)
                    and len(item) == 4
                    and all(isinstance(v, (int, float)) for v in item)
                ):
                    lines.append((float(item[0]), float(item[1]), float(item[2]), float(item[3])))
            out[crop][str(cat)] = lines
    return out


def line_cost(a: Line, b: Line) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]) + abs(a[3] - b[3])


def eval_crop_cat(gt_lines: List[Line], pr_lines: List[Line]) -> EvalStats:
    stats = EvalStats()
    n_gt = len(gt_lines)
    n_pr = len(pr_lines)

    if n_gt == 0 and n_pr == 0:
        return stats
    if n_gt == 0:
        stats.unmatched_pred = n_pr
        return stats
    if n_pr == 0:
        stats.unmatched_gt = n_gt
        return stats

    cost = [[line_cost(gt_lines[i], pr_lines[j]) for j in range(n_pr)] for i in range(n_gt)]
    pairs = hungarian_assign(cost)

    total_abs_xy = 0.0
    total_abs_y = 0.0
    count_xy = 0
    count_y = 0
    matched = 0

    used_gt = set()
    used_pr = set()
    for i, j in pairs:
        if i >= n_gt or j >= n_pr:
            continue
        if i in used_gt or j in used_pr:
            continue
        used_gt.add(i)
        used_pr.add(j)
        matched += 1

        g = gt_lines[i]
        p = pr_lines[j]
        total_abs_xy += abs(g[0] - p[0]) + abs(g[1] - p[1]) + abs(g[2] - p[2]) + abs(g[3] - p[3])
        count_xy += 4
        total_abs_y += abs(g[1] - p[1]) + abs(g[3] - p[3])
        count_y += 2

    stats.n_matched = matched
    stats.unmatched_gt = n_gt - matched
    stats.unmatched_pred = n_pr - matched
    stats.mae_xy = (total_abs_xy / count_xy) if count_xy else 0.0
    stats.mae_y = (total_abs_y / count_y) if count_y else 0.0
    return stats


def merge_stats(a: EvalStats, b: EvalStats) -> EvalStats:
    out = EvalStats()
    out.n_matched = a.n_matched + b.n_matched
    out.unmatched_gt = a.unmatched_gt + b.unmatched_gt
    out.unmatched_pred = a.unmatched_pred + b.unmatched_pred

    ax = a.n_matched * 4
    bx = b.n_matched * 4
    ay = a.n_matched * 2
    by = b.n_matched * 2

    out.mae_xy = ((a.mae_xy * ax + b.mae_xy * bx) / (ax + bx)) if (ax + bx) else 0.0
    out.mae_y = ((a.mae_y * ay + b.mae_y * by) / (ay + by)) if (ay + by) else 0.0
    return out


# -----------------------------
# Model callers
# -----------------------------
SYSTEM_PROMPT = "You are evaluating chart understanding. Return ONLY valid JSON. No markdown. No commentary."

USER_PROMPT = """Extract ALL straight line segments visible in the chart as start/end points.

Return ONLY valid JSON with this schema:

{
  "lines": {
    "<crop_name>": {
      "<category>": [[x0,y0,x1,y1], ...],
      ...
    },
    ...
  }
}

Rules:
- Use image pixel coordinates (origin top-left).
- crop_name can be ANY string you choose (e.g. "crop_1", "crop_2", ...). Use the same crop naming style as the image if it exists.
- If you cannot identify crops, put everything under a single crop key: "crop_all".
- Categories should be inferred by color groups; if unknown, use "unknown".
- x0,y0,x1,y1 must be numbers (floats ok).
- Output must be a single JSON object and nothing else.
"""


def call_openai_vision(image_b64_png: str) -> Dict[str, Any]:
    from openai import OpenAI  # type: ignore

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
    client = OpenAI(api_key=api_key)

    resp = client.responses.create(
        model=OPENAI_MODEL,
        temperature=TEMPERATURE,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": USER_PROMPT},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64_png}"},
                ],
            },
        ],
    )

    return safe_json_extract(resp.output_text)


def call_gemini_vision(image_b64_png: str) -> Dict[str, Any]:
    # pip install -U google-genai
    from google import genai
    from google.genai import types
    import base64
    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY (or hard-code it).")

    client = genai.Client(api_key=api_key)

    image_bytes = base64.b64decode(image_b64_png.encode("utf-8"))

    model_try_order = [GEMINI_MODEL, "gemini-2.0-flash", "gemini-2.5-flash"]
    last_err = None

    # Build parts ONCE (correct keyword arg usage)
    parts = [
        types.Part.from_text(text=SYSTEM_PROMPT),
        types.Part.from_text(text=USER_PROMPT),
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
    ]

    for m in model_try_order:
        try:
            resp = client.models.generate_content(
                model=m,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                ),
            )
            text = getattr(resp, "text", "") or ""
            return safe_json_extract(text)
        except Exception as e:
            last_err = e

    # If all fail, show available models to help choose correct model id
    try:
        available = []
        for mm in client.models.list():
            available.append(getattr(mm, "name", str(mm)))
        raise RuntimeError(
            f"Gemini failed for models {model_try_order}. Last error: {last_err}\n"
            f"Available models (sample): {available[:30]}"
        )
    except Exception:
        raise RuntimeError(f"Gemini failed for models {model_try_order}. Last error: {last_err}")


def count_lines(pred_lines_dict):
    total = 0
    for crop, cats in pred_lines_dict.items():
        for cat, arr in cats.items():
            total += len(arr)
    return total

# -----------------------------
# Evaluation
# -----------------------------
def flatten_lines(d: Dict[str, Dict[str, List[Line]]]) -> List[Line]:
    out = []
    for crop, cats in d.items():
        for cat, arr in cats.items():
            out.extend(arr)
    return out

def eval_one(gt: Dict[str, Any], pred: Dict[str, Any]) -> EvalStats:
    gtL = flatten_lines(gt_lines_only(gt))
    prL = flatten_lines(normalize_pred_to_lines(pred))
    return eval_crop_cat(gtL, prL)


def main():
    images_dir = Path(IMAGES_DIR)
    gt_dir = Path(GT_JSON_DIR)
    out_csv = Path(OUT_CSV)

    img_paths = sorted([p for p in images_dir.iterdir() if p.is_file() and is_image_file(p)])
    if MAX_IMAGES and MAX_IMAGES > 0:
        img_paths = img_paths[:MAX_IMAGES]

    if not img_paths:
        raise RuntimeError(f"No images found in: {images_dir}")

    rows = []
    total_openai = EvalStats()
    total_gemini = EvalStats()

    for i, img_path in enumerate(img_paths, start=1):
        stem = img_path.stem
        gt_path = gt_dir / f"{stem}.json"
        if not gt_path.exists():
            print(f"[WARN] Missing GT json for {img_path.name}: expected {gt_path.name}", file=sys.stderr)
            continue

        gt = load_json(gt_path)
        img_b64 = image_to_base64_png(img_path, max_side=MAX_SIDE)

        # OpenAI
        try:
            pred_oai = call_openai_vision(img_b64)
            stats_oai = eval_one(gt, pred_oai)
        except Exception as e:
            stats_oai = EvalStats()
            print(f"[ERROR] OpenAI failed on {img_path.name}: {e}", file=sys.stderr)

        try:
            pr_oai_lines = normalize_pred_to_lines(pred_oai)
            print("OAI extracted lines:", count_lines(pr_oai_lines))
            # print first 500 chars of raw text-like JSON
            print("OAI pred sample:", json.dumps(pred_oai)[:500])
        except Exception as e:
            print("OAI parse issue:", e)

        # Gemini
        try:
            pred_gem = call_gemini_vision(img_b64)
            stats_gem = eval_one(gt, pred_gem)
        except Exception as e:
            stats_gem = EvalStats()
            print(f"[ERROR] Gemini failed on {img_path.name}: {e}", file=sys.stderr)

        try:
            pr_gem_lines = normalize_pred_to_lines(pred_gem)
            print("GEM extracted lines:", count_lines(pr_gem_lines))
            print("GEM pred sample:", json.dumps(pred_gem)[:500])
        except Exception as e:
            print("GEM parse issue:", e)

        total_openai = merge_stats(total_openai, stats_oai)
        total_gemini = merge_stats(total_gemini, stats_gem)

        rows.append({
            "image": img_path.name,
            "openai_mae_xy": stats_oai.mae_xy,
            "openai_mae_y": stats_oai.mae_y,
            "openai_matched": stats_oai.n_matched,
            "openai_unmatched_gt": stats_oai.unmatched_gt,
            "openai_unmatched_pred": stats_oai.unmatched_pred,
            "gemini_mae_xy": stats_gem.mae_xy,
            "gemini_mae_y": stats_gem.mae_y,
            "gemini_matched": stats_gem.n_matched,
            "gemini_unmatched_gt": stats_gem.unmatched_gt,
            "gemini_unmatched_pred": stats_gem.unmatched_pred,
        })

        print(
            f"[{i}/{len(img_paths)}] {img_path.name} | "
            f"OAI mae_y={stats_oai.mae_y:.3f} matched={stats_oai.n_matched} | "
            f"GEM mae_y={stats_gem.mae_y:.3f} matched={stats_gem.n_matched}"
        )

    # Overall row
    rows.append({
        "image": "__OVERALL__",
        "openai_mae_xy": total_openai.mae_xy,
        "openai_mae_y": total_openai.mae_y,
        "openai_matched": total_openai.n_matched,
        "openai_unmatched_gt": total_openai.unmatched_gt,
        "openai_unmatched_pred": total_openai.unmatched_pred,
        "gemini_mae_xy": total_gemini.mae_xy,
        "gemini_mae_y": total_gemini.mae_y,
        "gemini_matched": total_gemini.n_matched,
        "gemini_unmatched_gt": total_gemini.unmatched_gt,
        "gemini_unmatched_pred": total_gemini.unmatched_pred,
    })

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("\nSaved:", str(out_csv))
    print(
        "OVERALL:",
        f"OpenAI mae_y={total_openai.mae_y:.4f}, mae_xy={total_openai.mae_xy:.4f}, matched={total_openai.n_matched} | "
        f"Gemini mae_y={total_gemini.mae_y:.4f}, mae_xy={total_gemini.mae_xy:.4f}, matched={total_gemini.n_matched}",
    )


if __name__ == "__main__":
    main()
