#!/usr/bin/env python3
"""
LLM Line Extraction Evaluation (ChatGPT vs Gemini) — complete script, no argparse.

What it does
- Reads images from IMAGES_DIR
- Reads GT jsons from GT_JSON_DIR (same filename stem)
- Calls:
  - OpenAI vision model (ChatGPT)
  - Gemini vision model (google-genai SDK)
- Parses model JSON: {"lines": { "<crop>": { "<cat>": [[x0,y0,x1,y1], ...] } } }
- Cleans predictions:
  - filter absurd coordinates
  - clip to image bounds
  - optional dedup
- Evaluates MAE with Hungarian matching:
  - default: GLOBAL matching (ignores crop/category names) — recommended because GT category keys are random IDs.
  - optional: STRICT matching per crop+category (set EVAL_MODE="strict")
- Writes results CSV.

Install
  pip install -U pillow openai google-genai
  # optional (better matching):
  pip install -U scipy numpy

IMPORTANT
- Hard-coding keys in the script is insecure. Do not commit this file.
"""


from PIL import ImageDraw, ImageFont

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
VIZ_DIR = Path("./viz")

OPENAI_MODEL = "gpt-4.1-mini"
GEMINI_MODEL = "gemini-1.5-pro"

TEMPERATURE = 0.0
MAX_SIDE = 1600
MAX_IMAGES = 0  # 0 = all

# Evaluation mode:
# - "global": ignore crop/category keys; match all GT lines to all predicted lines
# - "strict": match per crop+category key union
EVAL_MODE = "global"

# Prediction cleanup
ABSURD_FACTOR = 10.0   # drop line if any coord abs() > ABSURD_FACTOR * image_dim
DEDUP_TOL = 2.0        # pixels; set 0 to disable dedup
MAX_PRED_LINES = 800   # safety cap per model per image after cleanup (0 disables)

# Debug logging
PRINT_MODEL_JSON_SNIPPET = False
PRINT_GEMINI_RAW_ON_PARSE_FAIL = True
# =========================

OPENAI_API_KEY=""
GEMINI_API_KEY=""


# Put keys into env so the rest of the code can read them
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY


# -----------------------------
# Prompts (variable crops)
# -----------------------------
SYSTEM_PROMPT = "Return ONLY JSON. No markdown. No explanations. No extra keys."

USER_PROMPT_BASE = """Extract ALL straight line segments visible in the chart as start/end points.

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
- crop_name can be ANY string you choose (e.g. "crop_1", "crop_2", ...). If you cannot identify crops, use one crop key: "crop_all".
- Categories should be inferred by color groups; if unknown, use "unknown".
- x0,y0,x1,y1 must be numbers (floats ok).
Output must be a single JSON object and nothing else.
"""


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

    # existing
    mae_xy: float = 0.0      # over x0,y0,x1,y1
    mae_y: float = 0.0       # over y0,y1

    # new: start vs end
    mae_start: float = 0.0   # over (x0,y0)
    mae_end: float = 0.0     # over (x1,y1)
    mae_x0: float = 0.0
    mae_y0: float = 0.0
    mae_x1: float = 0.0
    mae_y1: float = 0.0

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


def image_to_base64_png(path: Path, max_side: int) -> Tuple[str, int, int]:
    """
    Returns (b64_png, width, height). Downscales if needed (maintains aspect).
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    w2, h2 = img.size

    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8"), w2, h2


def safe_json_extract(text: str) -> Dict[str, Any]:
    # direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # try to find a JSON object substring
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))


def normalize_pred_to_lines(pred: Dict[str, Any]) -> Dict[str, Dict[str, List[Line]]]:
    """
    Expect:
    { "lines": { "<crop>": { "<cat>": [[x0,y0,x1,y1], ...] } } }
    """
    if "lines" not in pred or not isinstance(pred["lines"], dict):
        raise ValueError("Prediction JSON must contain top-level key 'lines' as an object.")
    out: Dict[str, Dict[str, List[Line]]] = {}
    for crop, cats in pred["lines"].items():
        if not isinstance(cats, dict):
            continue
        out[str(crop)] = {}
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
            out[str(crop)][str(cat)] = lines
    return out


def gt_lines_only(gt: Dict[str, Any]) -> Dict[str, Dict[str, List[Line]]]:
    if "lines" not in gt or not isinstance(gt["lines"], dict):
        raise ValueError("GT JSON missing 'lines' object.")
    out: Dict[str, Dict[str, List[Line]]] = {}
    for crop, cats in gt["lines"].items():
        if not isinstance(cats, dict):
            continue
        out[str(crop)] = {}
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
            out[str(crop)][str(cat)] = lines
    return out


def flatten_lines_dict(d: Dict[str, Dict[str, List[Line]]]) -> List[Line]:
    out: List[Line] = []
    for crop, cats in d.items():
        for cat, arr in cats.items():
            out.extend(arr)
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

    used_gt, used_pr = set(), set()

    # sums
    sum_abs_x0 = sum_abs_y0 = sum_abs_x1 = sum_abs_y1 = 0.0
    matched = 0

    for i, j in pairs:
        if i >= n_gt or j >= n_pr:
            continue
        if i in used_gt or j in used_pr:
            continue
        used_gt.add(i)
        used_pr.add(j)
        matched += 1

        gx0, gy0, gx1, gy1 = gt_lines[i]
        px0, py0, px1, py1 = pr_lines[j]

        sum_abs_x0 += abs(gx0 - px0)
        sum_abs_y0 += abs(gy0 - py0)
        sum_abs_x1 += abs(gx1 - px1)
        sum_abs_y1 += abs(gy1 - py1)

    stats.n_matched = matched
    stats.unmatched_gt = n_gt - matched
    stats.unmatched_pred = n_pr - matched

    if matched > 0:
        # per-coordinate MAE
        stats.mae_x0 = sum_abs_x0 / matched
        stats.mae_y0 = sum_abs_y0 / matched
        stats.mae_x1 = sum_abs_x1 / matched
        stats.mae_y1 = sum_abs_y1 / matched

        # start point = (x0,y0), end point = (x1,y1)
        stats.mae_start = (sum_abs_x0 + sum_abs_y0) / (2 * matched)
        stats.mae_end = (sum_abs_x1 + sum_abs_y1) / (2 * matched)

        # existing combined metrics
        stats.mae_xy = (sum_abs_x0 + sum_abs_y0 + sum_abs_x1 + sum_abs_y1) / (4 * matched)
        stats.mae_y = (sum_abs_y0 + sum_abs_y1) / (2 * matched)

    return stats

def merge_stats(a: EvalStats, b: EvalStats) -> EvalStats:
    out = EvalStats()
    out.n_matched = a.n_matched + b.n_matched
    out.unmatched_gt = a.unmatched_gt + b.unmatched_gt
    out.unmatched_pred = a.unmatched_pred + b.unmatched_pred

    m = out.n_matched
    if m == 0:
        return out

    # weight by matched lines (not coords) for per-point/per-line MAEs
    def wavg(val_a, n_a, val_b, n_b):
        denom = n_a + n_b
        return (val_a * n_a + val_b * n_b) / denom if denom else 0.0

    out.mae_x0 = wavg(a.mae_x0, a.n_matched, b.mae_x0, b.n_matched)
    out.mae_y0 = wavg(a.mae_y0, a.n_matched, b.mae_y0, b.n_matched)
    out.mae_x1 = wavg(a.mae_x1, a.n_matched, b.mae_x1, b.n_matched)
    out.mae_y1 = wavg(a.mae_y1, a.n_matched, b.mae_y1, b.n_matched)

    out.mae_start = wavg(a.mae_start, a.n_matched, b.mae_start, b.n_matched)
    out.mae_end = wavg(a.mae_end, a.n_matched, b.mae_end, b.n_matched)

    # combined metrics are also per-matched-line averages, so same weighting works
    out.mae_xy = wavg(a.mae_xy, a.n_matched, b.mae_xy, b.n_matched)
    out.mae_y = wavg(a.mae_y, a.n_matched, b.mae_y, b.n_matched)

    return out



# -----------------------------
# Prediction cleanup
# -----------------------------
def clip_and_filter_lines(lines_dict: Dict[str, Dict[str, List[Line]]], img_w: int, img_h: int) -> Dict[str, Dict[str, List[Line]]]:
    out: Dict[str, Dict[str, List[Line]]] = {}
    wlim = float(img_w - 1)
    hlim = float(img_h - 1)
    for crop, cats in lines_dict.items():
        out[crop] = {}
        for cat, arr in cats.items():
            cleaned: List[Line] = []
            for (x0, y0, x1, y1) in arr:
                # drop absurd coords
                if (
                    abs(x0) > ABSURD_FACTOR * img_w or abs(x1) > ABSURD_FACTOR * img_w
                    or abs(y0) > ABSURD_FACTOR * img_h or abs(y1) > ABSURD_FACTOR * img_h
                ):
                    continue
                # clip
                x0 = max(0.0, min(wlim, float(x0)))
                x1 = max(0.0, min(wlim, float(x1)))
                y0 = max(0.0, min(hlim, float(y0)))
                y1 = max(0.0, min(hlim, float(y1)))
                cleaned.append((x0, y0, x1, y1))
            out[crop][cat] = cleaned
    return out


def dedup_lines(lines: List[Line], tol: float) -> List[Line]:
    if tol <= 0:
        return lines
    seen = set()
    out: List[Line] = []
    for (x0, y0, x1, y1) in lines:
        key = (round(x0 / tol), round(y0 / tol), round(x1 / tol), round(y1 / tol))
        if key in seen:
            continue
        seen.add(key)
        out.append((x0, y0, x1, y1))
    return out


def dedup_lines_dict(lines_dict: Dict[str, Dict[str, List[Line]]], tol: float) -> Dict[str, Dict[str, List[Line]]]:
    if tol <= 0:
        return lines_dict
    out: Dict[str, Dict[str, List[Line]]] = {}
    for crop, cats in lines_dict.items():
        out[crop] = {}
        for cat, arr in cats.items():
            out[crop][cat] = dedup_lines(arr, tol)
    return out


def cap_total_lines(lines_dict: Dict[str, Dict[str, List[Line]]], max_lines: int) -> Dict[str, Dict[str, List[Line]]]:
    if max_lines <= 0:
        return lines_dict
    flat = []
    for crop, cats in lines_dict.items():
        for cat, arr in cats.items():
            for line in arr:
                flat.append((crop, cat, line))
    if len(flat) <= max_lines:
        return lines_dict
    flat = flat[:max_lines]
    out: Dict[str, Dict[str, List[Line]]] = {}
    for crop, cat, line in flat:
        out.setdefault(crop, {}).setdefault(cat, []).append(line)
    return out


def count_lines(lines_dict: Dict[str, Dict[str, List[Line]]]) -> int:
    total = 0
    for crop, cats in lines_dict.items():
        for cat, arr in cats.items():
            total += len(arr)
    return total


# -----------------------------
# Model callers
# -----------------------------
def call_openai_vision(image_b64_png: str, user_prompt: str) -> Dict[str, Any]:
    from openai import OpenAI  # type: ignore

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")
    client = OpenAI(api_key=api_key)

    resp = client.responses.create(
        model=OPENAI_MODEL,
        temperature=TEMPERATURE,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64_png}"},
                ],
            },
        ],
    )
    return safe_json_extract(resp.output_text)


def call_gemini_vision(image_b64_png: str, user_prompt: str) -> Dict[str, Any]:
    """
    Requires: pip install -U google-genai
    Forces JSON output using response_mime_type.
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY.")

    client = genai.Client(api_key=api_key)
    image_bytes = base64.b64decode(image_b64_png.encode("utf-8"))

    # try a few common models; if your key doesn't have access, list_models at end
    model_try_order = [GEMINI_MODEL, "gemini-2.0-flash", "gemini-2.5-flash"]
    last_err = None

    parts = [
        types.Part.from_text(text=SYSTEM_PROMPT),
        types.Part.from_text(text=user_prompt),
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
    ]

    for m in model_try_order:
        try:
            resp = client.models.generate_content(
                model=m,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    response_mime_type="application/json",
                ),
            )
            text = getattr(resp, "text", "") or ""
            try:
                return safe_json_extract(text)
            except Exception as e:
                if PRINT_GEMINI_RAW_ON_PARSE_FAIL:
                    print(f"\n[Gemini RAW model={m}]\n{text[:2000]}\n", file=sys.stderr)
                raise e
        except Exception as e:
            last_err = e

    # If all fail, show some available models
    try:
        names = []
        for mm in client.models.list():
            names.append(getattr(mm, "name", str(mm)))
        raise RuntimeError(
            f"Gemini failed for models {model_try_order}. Last error: {last_err}\n"
            f"Available models (sample): {names[:30]}"
        )
    except Exception:
        raise RuntimeError(f"Gemini failed for models {model_try_order}. Last error: {last_err}")


# -----------------------------
# Evaluation wrappers
# -----------------------------
def eval_one_strict(gt: Dict[str, Any], pr: Dict[str, Dict[str, List[Line]]]) -> EvalStats:
    gtL = gt_lines_only(gt)
    overall = EvalStats()
    all_crops = set(gtL.keys()) | set(pr.keys())
    for crop in sorted(all_crops):
        gt_c = gtL.get(crop, {})
        pr_c = pr.get(crop, {})
        all_cats = set(gt_c.keys()) | set(pr_c.keys())
        for cat in sorted(all_cats):
            s = eval_crop_cat(gt_c.get(cat, []), pr_c.get(cat, []))
            overall = merge_stats(overall, s)
    return overall


def eval_one_global(gt: Dict[str, Any], pr: Dict[str, Dict[str, List[Line]]]) -> EvalStats:
    gt_flat = flatten_lines_dict(gt_lines_only(gt))
    pr_flat = flatten_lines_dict(pr)
    return eval_crop_cat(gt_flat, pr_flat)



# ----------------------------
# visualization
# ----------------------------
def draw_predictions_on_image(
    img_path: Path,
    lines_dict: Dict[str, Dict[str, List[Tuple[float, float, float, float]]]],
    out_path: Path,
    title: str = "",
    max_lines: int = 300,
    width: int = 2,
):
    """
    Draw predicted lines on the image and save it.
    Uses a simple category->color mapping (fallback to white).
    """
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Simple mapping for common names; unknown categories will be white
    cat_colors = {
        "blue": (0, 120, 255),
        "cyan": (0, 255, 255),
        "pink": (255, 0, 170),
        "magenta": (255, 0, 255),
        "red": (255, 0, 0),
        "green": (0, 200, 0),
        "yellow": (255, 215, 0),
        "orange": (255, 140, 0),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "unknown": (255, 255, 255),
    }

    # Flatten lines across crops/categories
    flat = []
    for crop, cats in lines_dict.items():
        for cat, arr in cats.items():
            for (x0, y0, x1, y1) in arr:
                flat.append((crop, cat, x0, y0, x1, y1))

    # Limit to avoid unreadable clutter
    if max_lines > 0:
        flat = flat[:max_lines]

    # Draw
    for crop, cat, x0, y0, x1, y1 in flat:
        c = cat_colors.get(cat.lower(), (255, 255, 255))
        draw.line((x0, y0, x1, y1), fill=c, width=width)

        # Optional: draw endpoints as small circles
        r = 2
        draw.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), outline=c, width=width)
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), outline=c, width=width)

    # Optional title text
    if title:
        try:
            draw.text((10, 10), title, fill=(255, 255, 255))
        except Exception:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)



# -----------------------------
# Main
# -----------------------------
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
    total_oai = EvalStats()
    total_gem = EvalStats()

    for idx, img_path in enumerate(img_paths, start=1):
        stem = img_path.stem
        gt_path = gt_dir / f"{stem}.json"
        if not gt_path.exists():
            print(f"[WARN] Missing GT json for {img_path.name}: expected {gt_path.name}", file=sys.stderr)
            continue

        gt = load_json(gt_path)

        # Encode image (possibly downscaled) and get its size (of encoded image)
        img_b64, img_w, img_h = image_to_base64_png(img_path, max_side=MAX_SIDE)

        # Build prompt; you can inject GT crop/category keys if you want stricter outputs.
        user_prompt = USER_PROMPT_BASE

        # ---------------- OpenAI ----------------
        pred_oai_raw = None
        pr_oai = {"crop_all": {"unknown": []}}  # default empty
        stats_oai = EvalStats(unmatched_gt=0, unmatched_pred=0)

        # After pr_oai is ready:
        draw_predictions_on_image(
            img_path=img_path,
            lines_dict=pr_oai,
            out_path=VIZ_DIR / f"{img_path.stem}_OAI.png",
            title=f"OAI ({img_path.name})",
            max_lines=300,
            width=2,
        )

        try:
            pred_oai_raw = call_openai_vision(img_b64, user_prompt=user_prompt)
            if PRINT_MODEL_JSON_SNIPPET:
                print("OAI pred sample:", json.dumps(pred_oai_raw)[:600])

            pr_oai = normalize_pred_to_lines(pred_oai_raw)
            pr_oai = clip_and_filter_lines(pr_oai, img_w, img_h)
            pr_oai = dedup_lines_dict(pr_oai, DEDUP_TOL)
            pr_oai = cap_total_lines(pr_oai, MAX_PRED_LINES)

            if EVAL_MODE.lower() == "strict":
                stats_oai = eval_one_strict(gt, pr_oai)
            else:
                stats_oai = eval_one_global(gt, pr_oai)

        except Exception as e:
            print(f"[ERROR] OpenAI failed on {img_path.name}: {e}", file=sys.stderr)

        # ---------------- Gemini ----------------
        pred_gem_raw = None
        pr_gem = {"crop_all": {"unknown": []}}  # default empty
        stats_gem = EvalStats(unmatched_gt=0, unmatched_pred=0)

        # After pr_gem is ready:
        draw_predictions_on_image(
            img_path=img_path,
            lines_dict=pr_gem,
            out_path=VIZ_DIR / f"{img_path.stem}_GEM.png",
            title=f"GEM ({img_path.name})",
            max_lines=300,
            width=2,
        )

        try:
            pred_gem_raw = call_gemini_vision(img_b64, user_prompt=user_prompt)
            if PRINT_MODEL_JSON_SNIPPET:
                print("GEM pred sample:", json.dumps(pred_gem_raw)[:600])

            pr_gem = normalize_pred_to_lines(pred_gem_raw)
            pr_gem = clip_and_filter_lines(pr_gem, img_w, img_h)
            pr_gem = dedup_lines_dict(pr_gem, DEDUP_TOL)
            pr_gem = cap_total_lines(pr_gem, MAX_PRED_LINES)

            if EVAL_MODE.lower() == "strict":
                stats_gem = eval_one_strict(gt, pr_gem)
            else:
                stats_gem = eval_one_global(gt, pr_gem)

        except Exception as e:
            print(f"[ERROR] Gemini failed on {img_path.name}: {e}", file=sys.stderr)

        # Aggregate totals (weighted by matched coords)
        total_oai = merge_stats(total_oai, stats_oai)
        total_gem = merge_stats(total_gem, stats_gem)

        # Optional counts
        oai_count = count_lines(pr_oai) if isinstance(pr_oai, dict) else 0
        gem_count = count_lines(pr_gem) if isinstance(pr_gem, dict) else 0
        gt_count = len(flatten_lines_dict(gt_lines_only(gt)))

        print(
            f"[{idx}/{len(img_paths)}] {img_path.name} | "
            f"GT={gt_count} | "
            f"OAI lines={oai_count} mae_y={stats_oai.mae_y:.3f} matched={stats_oai.n_matched} | "
            f"GEM lines={gem_count} mae_y={stats_gem.mae_y:.3f} matched={stats_gem.n_matched}"
        )

        rows.append({
            "image": img_path.name,
            "img_w": img_w,
            "img_h": img_h,
            "gt_lines": gt_count,

            "oai_pred_lines": oai_count,
            "oai_mae_xy": stats_oai.mae_xy,
            "oai_mae_y": stats_oai.mae_y,
            "oai_mae_start": stats_oai.mae_start,
            "oai_mae_end": stats_oai.mae_end,
            "oai_mae_x0": stats_oai.mae_x0,
            "oai_mae_y0": stats_oai.mae_y0,
            "oai_mae_x1": stats_oai.mae_x1,
            "oai_mae_y1": stats_oai.mae_y1,
            "oai_matched": stats_oai.n_matched,
            "oai_unmatched_gt": stats_oai.unmatched_gt,
            "oai_unmatched_pred": stats_oai.unmatched_pred,

            "gem_pred_lines": gem_count,
            "gem_mae_xy": stats_gem.mae_xy,
            "gem_mae_y": stats_gem.mae_y,
            "gem_mae_start": stats_gem.mae_start,
            "gem_mae_end": stats_gem.mae_end,
            "gem_mae_x0": stats_gem.mae_x0,
            "gem_mae_y0": stats_gem.mae_y0,
            "gem_mae_x1": stats_gem.mae_x1,
            "gem_mae_y1": stats_gem.mae_y1,
            "gem_matched": stats_gem.n_matched,
            "gem_unmatched_gt": stats_gem.unmatched_gt,
            "gem_unmatched_pred": stats_gem.unmatched_pred,
        })

    # Overall row
    rows.append({
        "image": "__OVERALL__",
        "img_w": "",
        "img_h": "",
        "gt_lines": "",
        "oai_pred_lines": "",
        "oai_mae_xy": total_oai.mae_xy,
        "oai_mae_y": total_oai.mae_y,
        "oai_matched": total_oai.n_matched,
        "oai_unmatched_gt": total_oai.unmatched_gt,
        "oai_unmatched_pred": total_oai.unmatched_pred,
        "gem_pred_lines": "",
        "gem_mae_xy": total_gem.mae_xy,
        "gem_mae_y": total_gem.mae_y,
        "gem_matched": total_gem.n_matched,
        "gem_unmatched_gt": total_gem.unmatched_gt,
        "gem_unmatched_pred": total_gem.unmatched_pred,
        "oai_mae_start": total_oai.mae_start,
        "oai_mae_end": total_oai.mae_end,
        "oai_mae_x0": total_oai.mae_x0,
        "oai_mae_y0": total_oai.mae_y0,
        "oai_mae_x1": total_oai.mae_x1,
        "oai_mae_y1": total_oai.mae_y1,

        "gem_mae_start": total_gem.mae_start,
        "gem_mae_end": total_gem.mae_end,
        "gem_mae_x0": total_gem.mae_x0,
        "gem_mae_y0": total_gem.mae_y0,
        "gem_mae_x1": total_gem.mae_x1,
        "gem_mae_y1": total_gem.mae_y1,

    })

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nSaved:", str(out_csv))
    print(
        "OVERALL:",
        f"OAI mae_y={total_oai.mae_y:.4f}, mae_xy={total_oai.mae_xy:.4f}, matched={total_oai.n_matched} | "
        f"GEM mae_y={total_gem.mae_y:.4f}, mae_xy={total_gem.mae_xy:.4f}, matched={total_gem.n_matched}"
    )


if __name__ == "__main__":
    main()
