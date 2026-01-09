
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

from pydantic import BaseModel, Field
# Gemini + structured parsing
from google import genai
from google.genai import types


# Your sAP metric (must be importable; keep sap_metric.py in same folder or PYTHONPATH)
from sap_metric import LineSegmentSAPMetric


# =========================
# Config (edit these)
# =========================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
project_root = Path(project_root)

IMAGE_DIR = project_root / "data/synthetic_plots/multi_cat/testing/m_crops/images_224"
GT_JSON_PATH = project_root / "data/synthetic_plots/multi_cat/testing/m_crops/test.json"


#
# IMAGE_DIR = project_root / "data/synthetic_plots/testing/images_100"
# GT_JSON_PATH = project_root / "data/synthetic_plots/testing/test.json"

OUT_CSV = project_root / "outputs/llms/results_Gemini_only_with_sap_test_mae_1k.csv"

# Gemini_MODEL = "gpt-4.1-mini"  # change if you want

USE_Gemini = True
MAX_IMAGES = 0
MAX_SIDE = 0
MAX_MATCH_COST = 400.0


GEMINI_API_KEY="AIzaSyDnmum8L8pHicm-8OqNOFzuTeG7fxI1yCM"
# GEMINI_API_KEY="AIzaSyBYu99uo9JfS0z_p-akz4XwJCN3qk8J-gc" # daniel's key
if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY


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
- Return ONLY JSON. No extra text.
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
def simple_line_mae(gt_lines: List[Line], pr_lines: List[Line]) -> float:
    """
    Simple MAE between GT and Pred lines (coordinate-wise).
    - Pairs by index (no matching).
    - Missing lines contribute full coordinate error (compared to 0).
    - Normalized by max(len(gt), len(pred)) and 4 coords per line.
    """
    G = len(gt_lines)
    P = len(pr_lines)
    N = max(G, P)
    if N == 0:
        return 0.0

    total_abs = 0.0
    M = min(G, P)

    # Paired lines
    for i in range(M):
        gx0, gy0, gx1, gy1 = gt_lines[i]
        px0, py0, px1, py1 = pr_lines[i]
        total_abs += (
            abs(gx0 - px0) + abs(gy0 - py0) +
            abs(gx1 - px1) + abs(gy1 - py1)
        )

    # Missing GT lines (pred assumed 0)
    for i in range(M, G):
        gx0, gy0, gx1, gy1 = gt_lines[i]
        total_abs += abs(gx0) + abs(gy0) + abs(gx1) + abs(gy1)

    # Missing Pred lines (gt assumed 0)
    for i in range(M, P):
        px0, py0, px1, py1 = pr_lines[i]
        total_abs += abs(px0) + abs(py0) + abs(px1) + abs(py1)

    return total_abs / (N * 4.0)


# =========================
# Helpers
# =========================

def safe_json_extract(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output")

    # direct
    try:
        return json.loads(text)
    except Exception:
        pass

    # dict block
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return json.loads(m.group(0))

    # list block
    m2 = re.search(r"\[[\s\S]*\]", text)
    if m2:
        return json.loads(m2.group(0))

    raise ValueError("No JSON found in model output")


def extract_lines_from_any_json(pred: Any) -> List[Line]:
    if isinstance(pred, list):
        raw = pred
    elif isinstance(pred, dict):
        raw = pred.get("lines", [])
    else:
        return []

    out: List[Line] = []
    for l in raw:
        if isinstance(l, (list, tuple)) and len(l) == 4:
            try:
                out.append((float(l[0]), float(l[1]), float(l[2]), float(l[3])))
            except Exception:
                pass
    return out

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
# Gemini structured output
# =========================
import random
import time

def call_with_retries(fn_call, *, max_retries=6, base_sleep=2.0):
    """
    Retries on transient Gemini errors:
      - 503 UNAVAILABLE (model overloaded)
      - 429 RESOURCE_EXHAUSTED / rate limits
    Exponential backoff + jitter.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn_call()
        except Exception as e:
            msg = str(e)

            transient = ("503" in msg or "UNAVAILABLE" in msg or
                         "429" in msg or "RESOURCE_EXHAUSTED" in msg)

            if not transient or attempt == max_retries:
                raise

            # If API suggests a retry delay, honor it (e.g., "retry in 41s")
            m = re.search(r"retry in ([0-9]+(\.[0-9]+)?)s", msg, re.IGNORECASE)
            if m:
                sleep_s = float(m.group(1))
            else:
                # exponential backoff with jitter
                sleep_s = base_sleep * (2 ** attempt)
                sleep_s = sleep_s * (0.7 + 0.6 * random.random())  # jitter 0.7–1.3

            print(f"[WARN] Gemini transient error, retrying in {sleep_s:.1f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(sleep_s)

    # unreachable
    raise RuntimeError("Retries exhausted")

def is_daily_quota_exhausted(err: Exception) -> bool:
    msg = (str(err) or "").lower()
    return (
        "daily quota" in msg and "exhaust" in msg
    ) or (
        "quota exceeded" in msg
    ) or (
        "exceeded your current quota" in msg
    )

class LinesOut(BaseModel):
    lines: List[List[float]] = Field(default_factory=list)

class GeminiLinePredictor:
    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        api_key: str | None = None,
        api_version: str = "v1",
        temperature: float = 0.0,
    ):
        self.model = model
        self.temperature = temperature

        http_options = types.HttpOptions(api_version=api_version)
        if api_key:
            self.client = genai.Client(api_key=api_key, http_options=http_options)
        else:
            self.client = genai.Client(http_options=http_options)

        # client_kwargs = {}
        #
        # # Some versions of google-genai do not have types.HttpOptions
        # try:
        #     http_options = types.HttpOptions(api_version=api_version)
        #     client_kwargs["http_options"] = http_options
        # except Exception:
        #     # Older SDK: just skip api_version override
        #     pass
        #
        # if api_key:
        #     self.client = genai.Client(api_key=api_key, **client_kwargs)
        # else:
        #     self.client = genai.Client(**client_kwargs)

    @staticmethod
    def _decode_b64_png(img_b64: str) -> bytes:
        return base64.b64decode(img_b64)

    @staticmethod
    def _normalize_lines(lines: List[List[float]]) -> List[Line]:
        out: List[Line] = []
        for l in lines or []:
            if isinstance(l, list) and len(l) == 4:
                try:
                    x0, y0, x1, y1 = map(float, l)
                    out.append((x0, y0, x1, y1))
                except Exception:
                    pass
        return out

    def predict_lines_from_b64png(self, img_b64: str, prompt: str) -> List[Line]:
        img_bytes = self._decode_b64_png(img_b64)

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                prompt,
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
            ],
            config=types.GenerateContentConfig(
                temperature=self.temperature,
            ),
        )

        text = getattr(response, "text", "") or ""
        pred = safe_json_extract(text)
        return extract_lines_from_any_json(pred)


        # With schema, SDK returns parsed output
        parsed = response.parsed  # should be LinesOut
        return self._normalize_lines(parsed.lines)

    def list_models(self, limit: int = 50) -> List[str]:
        """
        Optional helper to debug model-name 404s:
        prints available models from the SDK.
        """
        names = []
        # client.models.list() exists in the SDK reference (“List Base Models”). :contentReference[oaicite:6]{index=6}
        for i, m in enumerate(self.client.models.list()):
            if i >= limit:
                break
            if getattr(m, "name", None):
                names.append(m.name)
        return names


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
    tot_gem = MAEStats()
    # Global simple MAE accumulators
    global_simple_abs = 0.0
    global_simple_count = 0  # counts lines (not images)

    # CSV rows
    rows: List[Dict[str, Any]] = []

    fieldnames = [
        "image",
        "gt_lines",
        "gem_pred_lines",
        "gem_matched",
        "gem_unmatched_gt",
        "gem_unmatched_pred",
        "gem_mae_start",
        "gem_mae_end",
        "gem_mae_all",
        "gem_simple_mae",
        "sap5",
        "sap10",
        "sap15",
    ]

    # Open CSV early so we can save partial progress
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    f = out_csv.open("w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    f.flush()
    stop_early = False

    for k, img_path in enumerate(img_paths, start=1):
        fn = img_path.name
        gt_lines = gt_index.get(fn)
        if gt_lines is None:
            print(f"[WARN] No GT found for {fn}, skipping.", file=sys.stderr)
            continue

        img_b64 = image_to_base64_png(img_path, max_side=MAX_SIDE)

        # ---- Gemini preds ----
        gem_lines: List[Line] = []
        gem_stats = MAEStats(unmatched_gt=len(gt_lines), unmatched_pred=0)

        if USE_Gemini:
            try:
                gem = GeminiLinePredictor(
                    model="gemini-2.0-flash",  # example model from Google docs :contentReference[gemcite:7]{index=7}
                    api_key=GEMINI_API_KEY        # optional; else use env GEMINI_API_KEY
                )
                gem_lines = gem.predict_lines_from_b64png(img_b64, USER_PROMPT_BASE)
                gem_simple_mae = simple_line_mae(gt_lines, gem_lines)
                # ---- accumulate global simple MAE ----
                G = len(gt_lines)
                P = len(gem_lines)
                N = max(G, P)

                if N > 0:
                    global_simple_count += N
                    global_simple_abs += gem_simple_mae * (N * 4.0)

                # gem_lines = call_with_retries(
                #     lambda: gem.predict_lines_from_b64png(img_b64, USER_PROMPT_BASE)
                # )

                # if gem_lines:
                #     xs = [v for l in gem_lines for v in (l[0], l[2])]
                #     ys = [v for l in gem_lines for v in (l[1], l[3])]
                #     print(f"[DBG] {fn} pred x:[{min(xs):.2f},{max(xs):.2f}] y:[{min(ys):.2f},{max(ys):.2f}]")

                gem_stats = compute_mae_stats(gt_lines, gem_lines)

                # ---- sAP add_image ----
                pred_lines_yx = xyxy_list_to_yx_tensor(gem_lines)
                gt_lines_yx = xyxy_list_to_yx_tensor(gt_lines)
                pred_scores = scores_from_length(gem_lines)
                sap_metric.add_image(pred_lines_yx, pred_scores, gt_lines_yx)

            except Exception as e:
                print(f"[WARN] Gemini failed on {fn}: {e}", file=sys.stderr)
                if is_daily_quota_exhausted(e):
                    print("[STOP] Daily quota exhausted. Saving partial CSV and stopping.", file=sys.stderr)
                    stop_early = True

        # accumulate totals (MAE)
        tot_gem.matched += gem_stats.matched
        tot_gem.unmatched_gt += gem_stats.unmatched_gt
        tot_gem.unmatched_pred += gem_stats.unmatched_pred
        tot_gem._sum_abs_start += gem_stats._sum_abs_start
        tot_gem._sum_abs_end += gem_stats._sum_abs_end
        tot_gem._sum_abs_all += gem_stats._sum_abs_all

        row = {
            "image": fn,
            "gt_lines": len(gt_lines),
            "gem_pred_lines": len(gem_lines),
            "gem_matched": gem_stats.matched,
            "gem_unmatched_gt": gem_stats.unmatched_gt,
            "gem_unmatched_pred": gem_stats.unmatched_pred,
            "gem_mae_start": gem_stats.mae_start,
            "gem_mae_end": gem_stats.mae_end,
            "gem_mae_all": gem_stats.mae_all,
            "sap5": "",
            "sap10": "",
            "sap15": "",
        }
        w.writerow(row)
        f.flush()

        print(
            f"[{k}/{len(img_paths)}] {fn} | GT={len(gt_lines)} | "
            f"gem: pred={len(gem_lines)} matched={gem_stats.matched} "
            f"MAE(start/end/all)={gem_stats.mae_start:.3f}/{gem_stats.mae_end:.3f}/{gem_stats.mae_all:.3f} | "
        )

        time.sleep(0.15)
        if stop_early:
            break

    # finalize MAE totals
    tot_gem.finalize()
    # compute sAP totals
    sap = sap_metric.compute_sap()  # dict keyed by thresholds

    print("\n=============== TOTAL (MAE) ===============")
    print(
        f"matched={tot_gem.matched} | "
        f"MAE(start/end/all)={tot_gem.mae_start:.3f}/{tot_gem.mae_end:.3f}/{tot_gem.mae_all:.3f}"
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
        "gem_pred_lines",
        "gem_matched",
        "gem_unmatched_gt",
        "gem_unmatched_pred",
        "gem_mae_start",
        "gem_mae_end",
        "gem_mae_all",
        "gem_simple_mae",
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
                "gem_pred_lines": "",
                "gem_matched": tot_gem.matched,
                "gem_unmatched_gt": tot_gem.unmatched_gt,
                "gem_unmatched_pred": tot_gem.unmatched_pred,
                "gem_mae_start": tot_gem.mae_start,
                "gem_mae_end": tot_gem.mae_end,
                "gem_mae_all": tot_gem.mae_all,
                "sap5": sap.get(5.0, 0.0),
                "sap10": sap.get(10.0, 0.0),
                "sap15": sap.get(15.0, 0.0),
            }
        )
        f.flush()
        f.close()
        print(f"\nSaved CSV: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
