import os
import random
import re
import json
from pathlib import Path
from glob import glob
import altair as alt
import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path

from src.pc.plot_gen.plot_utils import normalize_column, generate_hsv_pool, hsv_to_rgb, create_ticks_labels, \
    normalize_column_reverse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
# -----------------------------
# Helper Functions
# -----------------------------
# this class can be used after producing the results using demo.py
# provide the helper method

# reference : demon_stitch_xy_naming
def extract_image_id(filename):
    match = re.match(r"(\d+)_crop_", filename)
    return match.group(1) if match else None

def extract_crop_number(path):
    # crop_<digits>_<token>
    match = re.search(r'crop_(\d+)(?=_[^_]+)', path)
    return int(match.group(1)) if match else -1

def extract_label(path):
    # capture the token immediately after crop_<n>_
    m = re.search(r'crop_\d+_([^_]+)', path)
    return m.group(1) if m else None

def get_latest_path(base_path):
    updated_path = base_path.replace(".csv", "_updated.csv")
    return updated_path if os.path.exists(updated_path) else base_path

# -----------------------------
# Step 1: JSON to CSV Conversion
# -----------------------------

# --- replace convert_json_to_csv with this ---
def convert_json_to_csv(json_dir):
    # Resolve path relative to this script so it works no matter the CWD
    base = (Path(__file__).parent / json_dir).resolve()
    print(f"[convert_json_to_csv] Looking for JSONs in: {base}")

    if not base.exists():
        print(f"⚠️ Path does not exist: {base}")
        return

    # Use rglob to include subfolders; fall back to glob if you truly only want top-level
    json_files = list(base.rglob("*.json"))
    print(f"[convert_json_to_csv] Found {len(json_files)} JSON file(s).")

    pattern = re.compile(r'crop_(\d+)_([^_]+)')

    for json_file in json_files:
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Failed to read {json_file}: {e}")
            continue

        lines = data.get("lines", [])
        match = pattern.search(json_file.stem)
        crop = match.group(1) if match else "unknown"
        label = match.group(2) if match else "unknown"

        rows = []
        for p1, p2 in lines:
            x1, y1 = p1
            x2, y2 = p2
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            rows.append({
                f"crop_{crop}_y1": round(y1, 2),
                f"crop_{crop}_y2": round(y2, 2),
                "label": label
            })
        if not rows:
            print(f"⚠️ No lines found in {json_file}, skipping CSV.")
            continue

        df = pd.DataFrame(rows)
        csv_path = json_file.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ JSON converted to CSV: {csv_path}")


def convert_json_to_csv1(json_dir):
    pattern = re.compile(r'crop_(\d+)_([^_]+)')
    json_dir = Path(json_dir)
    print(json_dir)

    for json_file in json_dir.glob("*.json"):
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)


        lines = data.get("lines", [])
        match = pattern.search(json_file.stem)
        crop = match.group(1) if match else "unknown"
        label = match.group(2) if match else "unknown"

        rows = []
        for p1, p2 in lines:
            x1, y1 = p1
            x2, y2 = p2
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1

            row = {
                f"crop_{crop}_y1": round(y1, 2),
                f"crop_{crop}_y2": round(y2, 2),
                "label": label
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = json_file.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ JSON converted to CSV: {csv_path}")

def stitch_parallel_coordinates_keep_all(file_paths, column_map, category_id=1,
                                         threshold=5.0, carry_forward=True):
    dfs = []
    for fp, (y1, y2) in zip(file_paths, column_map):
        df = pd.read_csv(fp)
        if df.empty:
            continue
        dfs.append(df.rename(columns={y1: "y1", y2: "y2"}))

    if not dfs:
        return pd.DataFrame()

    # init from crop1
    paths = [{"crop1": r["y1"], "crop2": r["y2"]} for _, r in dfs[0].iterrows()]

    for i in range(1, len(dfs)):
        candidates = dfs[i]
        new_paths = []

        for path in paths:
            last = path[f"crop{i+1}"]  # previous crop's y2

            # nearest candidate in next crop
            j = (candidates["y1"] - last).abs().argsort().iloc[0]
            best = candidates.iloc[j]
            dist = abs(best["y1"] - last)

            extended = path.copy()
            if dist <= threshold:
                extended[f"crop{i+2}"] = best["y2"]
            else:
                # do NOT drop — keep path continuous
                extended[f"crop{i+2}"] = last if carry_forward else np.nan

            new_paths.append(extended)

        paths = new_paths

    out = pd.DataFrame(paths)
    out["cat"] = category_id
    return out


def spread_duplicate_values(values, eps=0.01):
    """
    values: 1D array-like of floats
    returns: new array where exact-duplicate values are slightly spread:
      y, y+eps, y-eps, y+2eps, y-2eps, ...
    """
    v = np.asarray(values, dtype=float).copy()
    out = v.copy()

    # group by exact value
    uniq = {}
    for idx, val in enumerate(v):
        if not np.isfinite(val):
            continue
        uniq.setdefault(val, []).append(idx)

    for val, idxs in uniq.items():
        if len(idxs) <= 1:
            continue
        # symmetric offsets centered at 0
        # order: 0, +1, -1, +2, -2, ...
        offsets = [0.0]
        k = 1
        while len(offsets) < len(idxs):
            offsets.append(+k * eps)
            if len(offsets) < len(idxs):
                offsets.append(-k * eps)
            k += 1
        for i, idx in enumerate(idxs):
            out[idx] = val + offsets[i]

    return out

def stitch_parallel_coordinates(file_paths, column_map, category_id=1, threshold=3.0):
    dfs = []
    for fp, (y1, y2) in zip(file_paths, column_map):
        try:
            df = pd.read_csv(fp)
        except pd.errors.EmptyDataError:
            print(f"⚠️ Empty CSV skipped: {fp}")
            continue

        if df.empty:
            print(f"⚠️ CSV has no rows, skipped: {fp}")
            continue

        dfs.append(df.rename(columns={y1: "y1", y2: "y2"}))

    if not dfs:
        print(f"⚠️ No valid CSVs for category/label {category_id}, skipping stitching.")
        return pd.DataFrame()  # empty, caller should handle

    paths = [
        {"crop1": row["y1"], "crop2": row["y2"]}
        for _, row in dfs[0].iterrows()
    ]

    for i in range(1, len(dfs)):
        new_paths = []
        for path in paths:
            last_axis_val = path[f"crop{i + 1}"]
            candidates = dfs[i]
            closest = candidates.iloc[(candidates["y1"] - last_axis_val).abs().argsort()[:1]]
            if abs(closest["y1"].values[0] - last_axis_val) <= threshold:
                extended_path = path.copy()
                extended_path[f"crop{i + 2}"] = closest["y2"].values[0]
                new_paths.append(extended_path)
        paths = new_paths

    for path in paths:
        path["cat"] = category_id

    return pd.DataFrame(paths)


def process_all_categories(base_dir, image_id, threshold=10.0):
    base_dir = Path(base_dir)

    # Detect all categories available for the image
    all_crop_files = list(base_dir.glob(f"{image_id}_crop_*_cat_*.csv"))
    category_ids = sorted(set(
        int(re.search(rf"{image_id}_crop_\d+_cat_(\d+)\.csv", f.name).group(1))
        for f in all_crop_files
    ))

    for category in category_ids:
        print(f"🔄 Processing category {category}...")
        file_paths, column_map = get_crop_csvs_and_column_map(base_dir, image_id, category)
        if not file_paths:
            print(f"⚠️ No files found for category {category}")
            continue

        stitched_df = stitch_parallel_coordinates_keep_all(file_paths, column_map, category_id=category, threshold=threshold)

        # stitched_df = stitch_parallel_coordinates_keep_all(file_paths, column_map, category_id=category, threshold=threshold)
        stitched_df = stitch_keep_segments(file_paths, column_map, category_id=category,
                                           threshold=threshold)


        output_path = base_dir / f"stitched_category{category}.csv"
        stitched_df.to_csv(output_path, index=False)
        print(f"✅ Saved: {output_path}")

def combine_chained_csvs(base_dir, image_id):
    base_dir = Path(base_dir)

    # Match files like: stitched_category1.csv
    csv_files = sorted(
        base_dir.glob("stitched_category*.csv"),
        key=lambda f: int(re.search(r"stitched_category(\d+)\.csv", f.name).group(1))
    )

    combined_rows = []

    for csv_file in csv_files:
        # Extract category number from filename
        cat_num = re.search(r"stitched_category(\d+)\.csv", csv_file.name).group(1)
        df = pd.read_csv(csv_file)
        df["cat"] = int(cat_num)
        combined_rows.append(df)

    # Combine all into one DataFrame
    final_df = pd.concat(combined_rows, ignore_index=True)

    # Output combined CSV
    output_path = base_dir / f"{image_id}_combined.csv"
    final_df.to_csv(output_path, index=False)
    print(f"📦 Final combined saved: {output_path}")

def get_crop_csvs_and_column_map(base_dir, image_id, category):
    # 'category' is a free-form label now (e.g., BKxJsl), not an int
    base_dir = Path(base_dir)
    matched_files = []

    # image_<id>_crop_<cropnum>_<label>(_<anydigits>)?.csv
    pattern = re.compile(r".*?(\d+)_crop_(\d+)_([^_]+)(?:_\d+)?\.csv$")

    for f in base_dir.glob("*_crop_*_*.csv"):
        m = pattern.match(f.name)
        if m:
            file_image_id, crop_num, file_label = m.groups()
            if int(file_image_id) == int(image_id) and file_label == str(category):
                matched_files.append((f, int(crop_num)))

    matched_files.sort(key=lambda x: x[1])
    file_paths = [str(f[0]) for f in matched_files]
    column_map = [(f"crop_{num}_y1", f"crop_{num}_y2") for _, num in matched_files]
    return file_paths, column_map

def stitch_keep_segments1(file_paths, column_map, category_id=1, threshold=5.0):
    dfs = []
    for fp, (y1_col, y2_col) in zip(file_paths, column_map):
        try:
            df = pd.read_csv(fp)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue

        df = df.rename(columns={y1_col: "y1", y2_col: "y2"})[["y1", "y2"]].copy()
        df["y1"] = pd.to_numeric(df["y1"], errors="coerce")
        df["y2"] = pd.to_numeric(df["y2"], errors="coerce")
        df = df.dropna(subset=["y1", "y2"])
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    # seed from crop1
    paths = [{"crop1_y1": float(r["y1"]), "crop1_y2": float(r["y2"])}
             for _, r in dfs[0].iterrows()]

    # extend crops 2..N
    for i in range(1, len(dfs)):
        candidates = dfs[i]
        prev_end_key = f"crop{i}_y2"      # keep this (dummy end)
        next_y1_key  = f"crop{i+1}_y1"
        next_y2_key  = f"crop{i+1}_y2"

        new_paths = []
        for path in paths:
            prev_end = path[prev_end_key]
            # if we don't have a valid previous endpoint, we can't match forward
            if not np.isfinite(prev_end) or candidates.empty:
                extended[next_y1_key] = np.nan
                extended[next_y2_key] = np.nan
                new_paths.append(extended)
                continue

            diffs = (candidates["y1"] - prev_end).abs()
            diffs = diffs.dropna()  # <-- critical

            # if all diffs were NA, no match possible
            if diffs.empty:
                extended[next_y1_key] = np.nan
                extended[next_y2_key] = np.nan
                new_paths.append(extended)
                continue

            j = diffs.idxmin()  # keep it as index type (don’t cast to int)
            dist = float(diffs.loc[j])

            extended = path.copy()
            if dist <= threshold:
                best = candidates.loc[j]
                extended[next_y1_key] = float(best["y1"])
                extended[next_y2_key] = float(best["y2"])
            else:
                extended[next_y1_key] = np.nan
                extended[next_y2_key] = np.nan

            new_paths.append(extended)

        paths = new_paths

    out = pd.DataFrame(paths)
    out["cat"] = category_id
    return out


def _spread_duplicates_for_matching(values, eps=0.1):
    """
    Spread EXACT duplicates by tiny offsets, for matching only.
    Example: y, y, y -> y, y+eps, y-eps

    eps must be much smaller than threshold.
    """
    v = np.asarray(values, dtype=float)
    out = v.copy()

    buckets = {}
    for i, val in enumerate(v):
        if not np.isfinite(val):
            continue
        buckets.setdefault(val, []).append(i)

    for val, idxs in buckets.items():
        if len(idxs) <= 1:
            continue

        offsets = [0.0]
        k = 1
        while len(offsets) < len(idxs):
            offsets.append(+k * eps)
            if len(offsets) < len(idxs):
                offsets.append(-k * eps)
            k += 1

        for j, i in enumerate(idxs):
            out[i] = val + offsets[j]

    return out

def stitch_keep_segments(file_paths, column_map, category_id=1, threshold=5.0,
                         eps=0.1, max_reuse=1):
    """
    Keeps BOTH y1 and y2 per crop:
      crop1_y1,crop1_y2,crop2_y1,crop2_y2,...

    Matching uses prev crop END to next crop START:
      crop{i}_y2  ->  crop{i+1}_y1

    eps: tiny offset applied ONLY to duplicate prev_end values for matching (not stored).
    max_reuse: limit how many times the same candidate row can be matched in this step.
              None = unlimited (your old behavior).
              1 = close to one-to-one (reduces collapse a lot).
              2/3 = allow some branching.
    """
    dfs = []
    for fp, (y1_col, y2_col) in zip(file_paths, column_map):
        try:
            df = pd.read_csv(fp)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue

        df = df.rename(columns={y1_col: "y1", y2_col: "y2"})[["y1", "y2"]].copy()
        df["y1"] = pd.to_numeric(df["y1"], errors="coerce")
        df["y2"] = pd.to_numeric(df["y2"], errors="coerce")
        df = df.dropna(subset=["y1", "y2"]).reset_index(drop=True)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    # seed from crop1
    paths = [{"crop1_y1": float(r["y1"]), "crop1_y2": float(r["y2"])}
             for _, r in dfs[0].iterrows()]

    # extend crops 2..N
    for i in range(1, len(dfs)):
        candidates = dfs[i]
        prev_end_key = f"crop{i}_y2"
        next_y1_key  = f"crop{i+1}_y1"
        next_y2_key  = f"crop{i+1}_y2"

        # candidate arrays for speed & stability
        cand_y1 = candidates["y1"].to_numpy(dtype=float)
        cand_y2 = candidates["y2"].to_numpy(dtype=float)

        used_count = np.zeros(len(cand_y1), dtype=int) if max_reuse is not None else None

        # precompute prev ends and their matching-adjusted version (tie-breaker)
        prev_ends_true = np.array([p.get(prev_end_key, np.nan) for p in paths], dtype=float)
        prev_ends_match = _spread_duplicates_for_matching(prev_ends_true, eps=eps)

        new_paths = []
        for p_idx, path in enumerate(paths):
            extended = path.copy()  # <-- FIX: define before any early-continue

            prev_end = prev_ends_match[p_idx]

            # cannot match forward
            if (not np.isfinite(prev_end)) or (len(cand_y1) == 0):
                extended[next_y1_key] = np.nan
                extended[next_y2_key] = np.nan
                new_paths.append(extended)
                continue

            # distances to all candidate starts
            d = np.abs(cand_y1 - prev_end)

            # optionally avoid reusing the same next segment too many times
            if used_count is not None:
                d = d + (used_count >= max_reuse) * 1e9

            j = int(np.argmin(d))
            dist = float(d[j])

            # if np.isfinite(dist) and dist <= threshold:
            #     extended[next_y1_key] = float(cand_y1[j])
            #     extended[next_y2_key] = float(cand_y2[j])
            #     if used_count is not None:
            #         used_count[j] += 1
            # else:
            #     extended[next_y1_key] = np.nan
            #     extended[next_y2_key] = np.nan    # keep the line connected (carry forward previous endpoint)
            #

            # always take the nearest REAL candidate if it exists
            if np.isfinite(dist):
                extended[next_y1_key] = float(cand_y1[j])
                extended[next_y2_key] = float(cand_y2[j])

                # flag whether it was a "good" match
                extended[f"match_ok_{i + 1}"] = (dist <= threshold)
                extended[f"match_dist_{i + 1}"] = dist

                if used_count is not None:
                    used_count[j] += 1
            else:
                # only here we truly have nothing to connect to
                extended[next_y1_key] = np.nan
                extended[next_y2_key] = np.nan
                extended[f"match_ok_{i + 1}"] = False
                extended[f"match_dist_{i + 1}"] = np.nan

            new_paths.append(extended)

        paths = new_paths

    out = pd.DataFrame(paths)
    out["cat"] = category_id
    return out

def to_plot_chain_df(stitched_segments_df):
    """
    Plot columns for your generate_plot():
      crop1 = crop1_y1
      crop2 = crop1_y2
      crop3 = crop2_y2
      crop4 = crop3_y2
      ...
      cat  = cat
    """
    df = stitched_segments_df.copy()

    # detect number of crops from available crop*_y2 columns
    crop_nums = []
    for c in df.columns:
        m = re.match(r"crop(\d+)_y2$", c)
        if m:
            crop_nums.append(int(m.group(1)))
    if not crop_nums:
        return pd.DataFrame()

    n = max(crop_nums)

    out = pd.DataFrame()
    out["crop1"] = df["crop1_y1"]
    out["crop2"] = df["crop1_y2"]

    for k in range(2, n + 1):
        out[f"crop{k+1}"] = df.get(f"crop{k}_y2", np.nan)

    out["cat"] = df["cat"]
    return out




def spread_overlaps_for_plot(df, cat_col="cat", eps_ratio=0.002):
    """
    For each axis column (crop1..cropN), if multiple rows share the same value,
    add a tiny vertical offset so they become visible.

    eps_ratio is relative to that axis' data range (so it scales with your data).
    """
    out = df.copy()
    axis_cols = [c for c in out.columns if c != cat_col]

    for col in axis_cols:
        s = pd.to_numeric(out[col], errors="coerce")
        mask = s.notna()
        if mask.sum() < 2:
            continue

        # scale epsilon to the axis range
        vmin, vmax = float(s[mask].min()), float(s[mask].max())
        span = max(vmax - vmin, 1e-9)
        eps = span * eps_ratio

        # identify duplicates (within category so different cats don't push each other)
        grp = out.loc[mask, [cat_col, col]].copy()
        # count position within identical (cat, value) groups: 0,1,2,...
        k = grp.groupby([cat_col, col]).cumcount()

        # center offsets around 0: 0,1,2 ->  -1,0,+1 (approximately)
        # offset = (k - (n-1)/2) * eps ; we need n per group
        n = grp.groupby([cat_col, col])[col].transform("size")
        offset = (k - (n - 1) / 2.0) * eps

        out.loc[mask, col] = s[mask].values + offset.values

    return out

def process_all_categories_and_combine(base_dir, image_id, threshold=10.0):
    base_dir = Path(base_dir)
    all_crop_files = list(base_dir.glob("*_crop_*_*.csv"))

    pattern = re.compile(r".*?(\d+)_crop_\d+_([^_]+)(?:_\d+)?\.csv")  # capture label
    labels = sorted(set(
        m.group(2) for f in all_crop_files
        if (m := pattern.match(f.name)) and int(m.group(1)) == int(image_id)
    ))


    combined_rows = []

    for label in labels:
        print(f"🔄 Processing label {label}...")
        file_paths, column_map = get_crop_csvs_and_column_map(base_dir, image_id, label)
        if not file_paths:
            print(f"⚠️ No files found for label {label}")
            continue

        #stitched_df = stitch_parallel_coordinates_keep_all(file_paths, column_map, category_id=label, threshold=threshold)
        stitched_segments = stitch_keep_segments(
            file_paths, column_map, category_id=label, threshold=threshold
        )

        # optional: save debug stitched segments (keeps y1/y2 per crop)
        stitched_segments.to_csv(base_dir / f"stitched_{label}_segments.csv", index=False)

        # this is what you will combine + plot
        stitched_df = to_plot_chain_df(stitched_segments)
        # stitched_path = base_dir / f"stitched_{label}.csv"
        # stitched_df.to_csv(stitched_path, index=False)
        # print(f"✅ Saved: {stitched_path}")
        combined_rows.append(stitched_df)

    if combined_rows:
        final_df = pd.concat(combined_rows, ignore_index=True)
        output_path = base_dir / f"{image_id}_combined.csv"
        final_df.to_csv(output_path, index=False)
        print(f"📦 Final combined saved: {output_path}")
    else:
        print("⚠️ No labels stitched. Nothing to combine.")


#%--------------------generate plot from the combined csvs -------------------------%
def generate_plot(df, filename=None, background_value=255,
                  grid_on=False, show_ticks_labels=False, category_hsv_map=None,
                  save_png=False, svg_dir=None, do_extraction=False):
    column_names = sorted(list(df.columns)[:-1])
    color_column = df.columns[-1]
    
    for col in column_names:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    normalized_columns = [normalize_column_reverse(df, col) for col in column_names]
    unique_categories = sorted(df[color_column].unique())

    if category_hsv_map is None:
        hsv_pool = generate_hsv_pool(30)
        selected_indices = np.random.choice(len(hsv_pool), len(unique_categories), replace=False)
        selected_hsvs = [hsv_pool[i] for i in selected_indices]
        category_hsv_map = dict(zip(unique_categories, selected_hsvs))
    category_colors = {
        category: hsv_to_rgb(hsv['h'], hsv['s'], hsv['v'])
        for category, hsv in category_hsv_map.items()
    }

    # altair chart components: base, lines, rules,
    base = alt.Chart(df).transform_window(index="count()").transform_fold(
        normalized_columns
    ).transform_calculate(
        mid="(datum.value + datum.value) / 2"
    ).properties(
        width=600,
        height=300
    )

    lines = base.mark_line(opacity=0.4).encode(
        x=alt.X('key:N', axis=alt.Axis(
            title=None,
            domain=False,
            labels=show_ticks_labels,
            labelAngle=0,
            ticks=False)),
        y=alt.Y('value:Q', axis=alt.Axis(
            title=None,
            domain=False,
            labels=False,
            ticks=False,
            grid=grid_on)),
        color=alt.Color(f"{color_column}:N", scale=alt.Scale(
            domain=list(category_colors.keys()),
            range=['rgb({},{},{})'.format(*color) for color in category_colors.values()]
        ), legend=None),
        detail="index:N",
        tooltip=column_names
    )

    palette10 = [
        "#BD081C",  # red
        "#0061FF",  # blue
        "#25D366",  # green
        "#FF5700",  # orange
        "#f781bf",  # pink
        "#8E44AD",  # magenta/purple

    ]

    # shuffle a copy so original palette stays intact
    shuffled_palette = palette10.copy()
    random.shuffle(shuffled_palette)

    # cycle if categories > 10
    range_ = [shuffled_palette[i % len(shuffled_palette)]
              for i in range(len(unique_categories))]

    lines = base.mark_line(opacity=0.8).encode(
        x=alt.X('key:N', axis=alt.Axis(title=None, domain=False, labels=show_ticks_labels, labelAngle=0, ticks=False)),
        y=alt.Y('value:Q', axis=alt.Axis(title=None, domain=False, labels=False, ticks=False, grid=grid_on)),
        color=alt.Color(
            f"{color_column}:N",
            scale=alt.Scale(
                domain=unique_categories,
                range=range_
            ),
            legend=None
        ),
        detail="index:N",
        tooltip=column_names
    )

    rules = base.mark_rule(color="#ccc", tooltip=None).encode(
        x=alt.X('key:N', axis=alt.Axis(title=None, labels=False, ticks=False))
    )

    tick_dfs = [create_ticks_labels(df, norm_col, orig_col)
                for norm_col, orig_col in zip(normalized_columns, column_names)]
    ticks_labels_df = pd.concat(tick_dfs)

    if show_ticks_labels:
        ticks = alt.Chart(ticks_labels_df).mark_tick(size=8, color="#ccc", orient="horizontal").encode(
            x='variable:N', y='value:Q')
        labels = alt.Chart(ticks_labels_df).mark_text(
            align='center', baseline='middle', dx=-10).encode(
            x='variable:N', y='value:Q', text='label:N')
    else:
        ticks = alt.Chart(ticks_labels_df).mark_tick(size=8, opacity=0, orient="horizontal").encode(
            x='variable:N', y='value:Q')
        labels = alt.Chart(ticks_labels_df).mark_text(
            align='center', baseline='middle', dx=-10, opacity=0).encode(
            x='variable:N', y='value:Q', text='label:N')

    # Axis configuration
    axis_config_x = {
        "domain": False,
        "labelAngle": 0,
        "title": None,
        "tickColor": "#ccc" if show_ticks_labels else "transparent",
        "labelColor": "#000" if show_ticks_labels else "transparent",
        "grid": grid_on,
        "gridColor": "#ccc" if grid_on else "transparent"
    }

    axis_config_y = axis_config_x.copy()

    chart = alt.layer(lines, rules, ticks, labels).configure_axisX(
        **axis_config_x
    ).configure_axisY(
        **axis_config_y
    ).configure_view(
        stroke=None
    )

    if background_value is not None:
        if isinstance(background_value, (tuple, list)) and len(background_value) == 3:
            r, g, b = background_value
            background_rgb = f"rgb({r},{g},{b})"
        else:  # fallback for legacy numeric values
            v = int(background_value)
            background_rgb = f"rgb({v},{v},{v})"
        chart = chart.configure(background=background_rgb)

    base, _ = os.path.splitext(filename)
    basename = os.path.basename(base)
    output_dir = os.path.dirname(filename)

    if svg_dir:
        os.makedirs(svg_dir, exist_ok=True)
        svg_filename = os.path.join(svg_dir, basename + ".svg")
    else:
        svg_filename = base + ".svg"

    png_filename = os.path.join(output_dir, basename + ".png")
    chart.save(svg_filename)
    if save_png:
        chart.save(png_filename, format="png")
    return chart

def main(base_dir):
    convert_json_to_csv(base_dir)

    csv_files = glob(os.path.join(base_dir, "*_crop_*_*.csv"))
    pattern = re.compile(r".*?(\d+)_crop_\d+_[^_]+(?:_\d+)?\.csv")

    image_ids = sorted(set(
        int(match.group(1)) for f in csv_files
        if (match := pattern.match(os.path.basename(f)))
    ))

    for image_id in image_ids:
        print(f"\n🚀 Starting pipeline for image ID: {image_id}")
        process_all_categories_and_combine(base_dir, image_id=image_id, threshold=10.0)

def build_category_hsv_map_from_dominant(df, image_id, dominant_colors):
    """
    dominant_colors JSON expected format:
    {
      "78.png": {
        "cat1": {"h": 0.95, "s": 1, "v": 1},
        "cat2": {"h": 0.51, "s": 1, "v": 1}
      },
      ...
    }
    """
    color_column = df.columns[-1]
    categories = sorted(df[color_column].unique())

    # Try to find matching key in dominant_colors JSON
    possible_keys = [f"{image_id}.png", f"r{image_id}.png"]
    cat_dict = None
    for key in possible_keys:
        if key in dominant_colors:
            cat_dict = dominant_colors[key]
            break

    if not cat_dict or not isinstance(cat_dict, dict):
        print(f"⚠️ No dominant colors found for image_id={image_id} in JSON.")
        return None

    # Sort cat1,cat2,... in numeric order
    def cat_sort_key(k):
        m = re.match(r"cat(\d+)$", str(k))
        return int(m.group(1)) if m else 10**9

    ordered_keys = sorted(cat_dict.keys(), key=cat_sort_key)

    # Build HSV list (already 0..1). Clamp just in case.
    hsv_list = []
    for k in ordered_keys:
        hsv = cat_dict.get(k, {})
        if not isinstance(hsv, dict):
            continue
        h = float(hsv.get("h", 0.0)) % 1.0
        s = max(0.0, min(1.0, float(hsv.get("s", 1.0))))
        v = max(0.0, min(1.0, float(hsv.get("v", 1.0))))
        hsv_list.append({"h": h, "s": s, "v": v})

    if not hsv_list:
        return None

    # Map dataframe categories -> HSV colors (cycle if more categories than colors)
    category_hsv_map = {}
    for i, cat in enumerate(categories):
        category_hsv_map[cat] = hsv_list[i % len(hsv_list)]

    return category_hsv_map

def find_image_json(image_id: str, json_root_dir, recursive: bool = True):
    """
    Looks for a file named image_<image_id>.json under json_root_dir.
    Returns a Path or None.
    """
    if image_id is None or image_id == "Unknown":
        return None

    json_root_dir = Path(json_root_dir)
    target_name = f"image_{image_id}.json"

    if recursive:
        matches = list(json_root_dir.rglob(target_name))
    else:
        matches = list(json_root_dir.glob(target_name))

    if not matches:
        return None

    # If multiple found, pick the first deterministically (sorted path)
    return sorted(matches)[0]


def load_json_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_category_hsv_map_from_dominant_old(df, image_id, dominant_colors):
    """
    df: combined dataframe (last column is category label)
    image_id: string from '<image_id>_combined.csv'
    dominant_colors: dict loaded from dominant_colors.json
                     keys like '261.png' or 'r261.png',
                     values = list of [Hdeg, S%, V%]
    """
    color_column = df.columns[-1]
    categories = sorted(df[color_column].unique())

    # Try to find a matching key in the JSON
    possible_keys = [f"{image_id}.png", f"r{image_id}.png"]
    hsv_triplets = None
    for key in possible_keys:
        if key in dominant_colors:
            hsv_triplets = dominant_colors[key]
            break

    if not hsv_triplets:
        print(f"⚠️ No dominant colors found for image_id={image_id} in JSON.")
        return None

    # JSON values are [Hdeg, S%, V%]; convert to 0–1 range
    normalized_hsvs = []
    for Hdeg, Sperc, Vperc in hsv_triplets:
        h = (Hdeg % 360.0) / 360.0
        s = max(0.0, min(1.0, Sperc / 100.0))
        v = max(0.0, min(1.0, Vperc / 100.0))
        normalized_hsvs.append({"h": h, "s": s, "v": v})

    if not normalized_hsvs:
        return None

    # Map categories → colors (cycle if there are more categories than colors)
    category_hsv_map = {}
    for i, cat in enumerate(categories):
        hsv = normalized_hsvs[i % len(normalized_hsvs)]
        category_hsv_map[cat] = hsv

    return category_hsv_map
def build_category_hsv_map_from_category_colors_json(df, category_colors_config):
    """
    category_colors_config can be:
      - a dict already loaded from JSON
      - OR a path to a JSON file

    Expected JSON format (like image_2.json):
    {
      "filename": "...",
      "category_colors": {
        "SomeCategoryName": {"h": 0.33, "s": 1, "v": 1},
        "OtherCategory":    {"h": 0.23, "s": 1, "v": 1}
      }
    }
    """
    # Load JSON if a filepath was passed
    if isinstance(category_colors_config, (str, Path)):
        with open(category_colors_config, "r", encoding="utf-8") as f:
            category_colors_config = json.load(f)

    if not isinstance(category_colors_config, dict):
        return None

    # Pull out category_colors (your requested "category_color is the item")
    cat_dict = category_colors_config.get("category_colors")
    if not isinstance(cat_dict, dict) or not cat_dict:
        return None

    color_column = df.columns[-1]
    categories = sorted(df[color_column].unique())

    # Build mapping for categories that exist in df.
    # If some df categories are missing from JSON, we’ll just skip them (plot code can fallback if you want).
    category_hsv_map = {}

    for cat in categories:
        hsv = cat_dict.get(str(cat))
        if isinstance(hsv, dict):
            h = float(hsv.get("h", 0.0)) % 1.0
            s = max(0.0, min(1.0, float(hsv.get("s", 1.0))))
            v = max(0.0, min(1.0, float(hsv.get("v", 1.0))))
            category_hsv_map[cat] = {"h": h, "s": s, "v": v}

    return category_hsv_map if category_hsv_map else None



def generate_plot_new(
    df,
    filename=None,
    background_value=255,
    grid_on=False,
    show_ticks_labels=False,
    category_hsv_map=None,
    save_png=False,
    svg_dir=None,
    do_extraction=False,   # kept for compatibility; not used
    # NEW:
    width=600,
    height=300,
    y_pad_ratio=0.25,      # add extra space above & below (fraction of data span)
    y_domain=None,         # override: [ymin, ymax]
    png_scale_factor=2     # sharper PNG if save_png=True
):
    """
    Parallel-coordinates style plot.

    - Set width/height to match your input image size.
    - Use y_pad_ratio to add similar vertical headroom/footroom as the original image.
    - Or pass y_domain=[lo,hi] to fully control y-scale.
    """

    # --- columns ---
    column_names = sorted(list(df.columns)[:-1])
    color_column = df.columns[-1]

    # ensure numeric
    for col in column_names:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # normalize columns (your existing helper)
    # normalized_columns = [normalize_column_reverse(df, col) for col in column_names]
    # global min/max across all axes (so none gets stretched)
    all_vals = pd.concat([pd.to_numeric(df[c], errors="coerce") for c in column_names], axis=0)
    gmin = float(all_vals.min())
    gmax = float(all_vals.max())
    span = max(gmax - gmin, 1e-9)

    # create global-normalized columns
    normalized_columns = []
    for col in column_names:
        ncol = f"{col}__gnorm"
        df[ncol] = (pd.to_numeric(df[col], errors="coerce") - gmin) / span
        df[ncol] = 1.0 - df[ncol]

        normalized_columns.append(ncol)

    # --- colors ---
    unique_categories = sorted(df[color_column].dropna().unique())

    if category_hsv_map is None:
        hsv_pool = generate_hsv_pool(30)
        selected_indices = np.random.choice(len(hsv_pool), len(unique_categories), replace=False)
        selected_hsvs = [hsv_pool[i] for i in selected_indices]
        category_hsv_map = dict(zip(unique_categories, selected_hsvs))

    category_colors = {
        category: hsv_to_rgb(hsv['h'], hsv['s'], hsv['v'])
        for category, hsv in category_hsv_map.items()
    }

    # If y_domain passed, use it as-is.
    if y_domain is None:
        vals = []
        for nc in normalized_columns:
            if nc in df.columns:
                s = pd.to_numeric(df[nc], errors="coerce")
                vals.append(s)
        if vals:
            allv = pd.concat(vals, ignore_index=True)
            vmin = float(allv.min())
            vmax = float(allv.max())
            span = max(vmax - vmin, 1e-9)
            y_lo = vmin - span * float(y_pad_ratio)
            y_hi = vmax + span * float(y_pad_ratio)
            y_domain = [y_lo, y_hi]
        else:
            y_domain = [0, 1]

    # --- base chart ---
    base = (
        alt.Chart(df)
        .transform_window(index="count()")
        .transform_fold(normalized_columns)
        .transform_calculate(mid="(datum.value + datum.value) / 2")
        .properties(width=width, height=height)
    )

    # --- lines ---
    lines = base.mark_line(opacity=0.6).encode(
        x=alt.X(
            "key:N",
            axis=alt.Axis(
                title=None,
                domain=False,
                labels=show_ticks_labels,
                labelAngle=0,
                ticks=False
            )
        ),
        y=alt.Y(
            "value:Q",
            scale=alt.Scale(domain=[y_domain[1], y_domain[0]]),
            axis=alt.Axis(
                title=None,
                domain=False,
                labels=False,
                ticks=False,
                grid=grid_on
            )
        ),
        color=alt.Color(
            f"{color_column}:N",
            scale=alt.Scale(
                domain=list(category_colors.keys()),
                range=["rgb({},{},{})".format(*rgb) for rgb in category_colors.values()]
            ),
            legend=None
        ),
        detail="index:N",
        tooltip=column_names
    )

    # --- vertical rules at each axis ---
    rules = base.mark_rule(color="#ccc", tooltip=None).encode(
        x=alt.X("key:N", axis=alt.Axis(title=None, labels=False, ticks=False))
    )

    # --- ticks/labels ---
    tick_dfs = [
        create_ticks_labels(df, norm_col, orig_col)
        for norm_col, orig_col in zip(normalized_columns, column_names)
    ]
    ticks_labels_df = pd.concat(tick_dfs, ignore_index=True)

    if show_ticks_labels:
        ticks = alt.Chart(ticks_labels_df).mark_tick(
            size=8, color="#ccc", orient="horizontal"
        ).encode(x="variable:N", y=alt.Y("value:Q", scale=alt.Scale(domain=y_domain)))

        labels = alt.Chart(ticks_labels_df).mark_text(
            align="center", baseline="middle", dx=-10
        ).encode(x="variable:N", y=alt.Y("value:Q", scale=alt.Scale(domain=y_domain)), text="label:N")
    else:
        ticks = alt.Chart(ticks_labels_df).mark_tick(
            size=8, opacity=0, orient="horizontal"
        ).encode(x="variable:N", y=alt.Y("value:Q", scale=alt.Scale(domain=y_domain)))

        labels = alt.Chart(ticks_labels_df).mark_text(
            align="center", baseline="middle", dx=-10, opacity=0
        ).encode(x="variable:N", y=alt.Y("value:Q", scale=alt.Scale(domain=y_domain)), text="label:N")

    # --- axis config ---
    axis_config_x = {
        "domain": False,
        "labelAngle": 0,
        "title": None,
        "tickColor": "#ccc" if show_ticks_labels else "transparent",
        "labelColor": "#000" if show_ticks_labels else "transparent",
        "grid": grid_on,
        "gridColor": "#ccc" if grid_on else "transparent",
    }
    axis_config_y = axis_config_x.copy()

    chart = (
        alt.layer(lines, rules, ticks, labels)
        .configure_axisX(**axis_config_x)
        .configure_axisY(**axis_config_y)
        .configure_view(stroke=None)
    )

    # --- background ---
    if background_value is not None:
        if isinstance(background_value, (tuple, list)) and len(background_value) == 3:
            r, g, b = background_value
            background_rgb = f"rgb({r},{g},{b})"
        else:
            v = int(background_value)
            background_rgb = f"rgb({v},{v},{v})"
        chart = chart.configure(background=background_rgb)

    # --- save ---
    if filename is not None:
        base_path, _ = os.path.splitext(filename)
        basename = os.path.basename(base_path)
        output_dir = os.path.dirname(filename)

        if svg_dir:
            os.makedirs(svg_dir, exist_ok=True)
            svg_filename = os.path.join(svg_dir, basename + ".svg")
        else:
            svg_filename = base_path + ".svg"

        png_filename = os.path.join(output_dir, basename + ".png")

        chart.save(svg_filename)
        if save_png:
            chart.save(png_filename, format="png", scale_factor=png_scale_factor)

    return chart

def generate_plot_newColor(
    df,
    filename=None,
    background_value=255,
    grid_on=False,
    show_ticks_labels=False,
    category_hsv_map=None,
    save_png=False,
    svg_dir=None,
    do_extraction=False,   # kept for compatibility; not used
    # NEW:
    width=600,
    height=300,
    y_pad_ratio=0.25,      # add extra space above & below (fraction of data span)
    y_domain=None,         # override: [ymin, ymax]
    png_scale_factor=2     # sharper PNG if save_png=True
):
    """
    Parallel-coordinates style plot.

    - Set width/height to match your input image size.
    - Use y_pad_ratio to add similar vertical headroom/footroom as the original image.
    - Or pass y_domain=[lo,hi] to fully control y-scale.
    """

    # --- columns ---
    column_names = sorted(list(df.columns)[:-1])
    color_column = df.columns[-1]

    # ensure numeric
    for col in column_names:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # normalize columns (your existing helper)
    # normalized_columns = [normalize_column_reverse(df, col) for col in column_names]
    # global min/max across all axes (so none gets stretched)
    all_vals = pd.concat([pd.to_numeric(df[c], errors="coerce") for c in column_names], axis=0)
    gmin = float(all_vals.min())
    gmax = float(all_vals.max())
    span = max(gmax - gmin, 1e-9)

    # create global-normalized columns
    normalized_columns = []
    for col in column_names:
        ncol = f"{col}__gnorm"
        df[ncol] = (pd.to_numeric(df[col], errors="coerce") - gmin) / span
        df[ncol] = 1.0 - df[ncol]

        normalized_columns.append(ncol)

    # --- colors ---
    unique_categories = sorted(df[color_column].dropna().unique())
    #
    # if category_hsv_map is None:
    #     hsv_pool = generate_hsv_pool(30)
    #     selected_indices = np.random.choice(len(hsv_pool), len(unique_categories), replace=False)
    #     selected_hsvs = [hsv_pool[i] for i in selected_indices]
    #     category_hsv_map = dict(zip(unique_categories, selected_hsvs))
    #
    # category_colors = {
    #     category: hsv_to_rgb(hsv['h'], hsv['s'], hsv['v'])
    #     for category, hsv in category_hsv_map.items()
    # }

    palette10 = [
        "#BD081C",  # red
        "#0061FF",  # blue
        "#25D366",  # green
        "#FF5700",  # orange
        "#f781bf",  # pink
        "#8E44AD",  # magenta/purple
        "#27AE60",

    ]

    # shuffle a copy so original palette stays intact
    shuffled_palette = palette10.copy()
    random.shuffle(shuffled_palette)

    # cycle if categories > 10
    range_ = [shuffled_palette[i % len(shuffled_palette)]
              for i in range(len(unique_categories))]

    # lines = base.mark_line(opacity=0.8).encode(
    #     x=alt.X('key:N', axis=alt.Axis(title=None, domain=False, labels=show_ticks_labels, labelAngle=0, ticks=False)),
    #     y=alt.Y('value:Q', axis=alt.Axis(title=None, domain=False, labels=False, ticks=False, grid=grid_on)),
    #     color=alt.Color(
    #         f"{color_column}:N",
    #         scale=alt.Scale(
    #             domain=unique_categories,
    #             range=range_
    #         ),
    #         legend=None
    #     ),
    #     detail="index:N",
    #     tooltip=column_names
    # )

    # --- compute Y domain for similar up/down spacing ---
    # Work in the "value" space (normalized columns) because that's what we plot.
    # We compute min/max across ALL normalized columns.
    # If y_domain passed, use it as-is.
    if y_domain is None:
        vals = []
        for nc in normalized_columns:
            if nc in df.columns:
                s = pd.to_numeric(df[nc], errors="coerce")
                vals.append(s)
        if vals:
            allv = pd.concat(vals, ignore_index=True)
            vmin = float(allv.min())
            vmax = float(allv.max())
            span = max(vmax - vmin, 1e-9)
            y_lo = vmin - span * float(y_pad_ratio)
            y_hi = vmax + span * float(y_pad_ratio)
            y_domain = [y_lo, y_hi]
        else:
            y_domain = [0, 1]

    # --- base chart ---
    base = (
        alt.Chart(df)
        .transform_window(index="count()")
        .transform_fold(normalized_columns)
        .transform_calculate(mid="(datum.value + datum.value) / 2")
        .properties(width=width, height=height)
    )

    # --- lines ---
    lines = base.mark_line(opacity=0.6).encode(
        x=alt.X(
            "key:N",
            axis=alt.Axis(
                title=None,
                domain=False,
                labels=show_ticks_labels,
                labelAngle=0,
                ticks=False
            )
        ),
        y=alt.Y(
            "value:Q",
            scale=alt.Scale(domain=[y_domain[1], y_domain[0]]),
            axis=alt.Axis(
                title=None,
                domain=False,
                labels=False,
                ticks=False,
                grid=grid_on
            )
        ),
        # color=alt.Color(
        #     f"{color_column}:N",
        #     scale=alt.Scale(
        #         domain=list(category_colors.keys()),
        #         range=["rgb({},{},{})".format(*rgb) for rgb in category_colors.values()]
        #     ),
        #     legend=None
        # ),
        color=alt.Color(
            f"{color_column}:N",
            scale=alt.Scale(
                domain=unique_categories,
                range=range_
            ),
            legend=None
        ),
        detail="index:N",
        tooltip=column_names
    )

    # --- vertical rules at each axis ---
    rules = base.mark_rule(color="#ccc", tooltip=None).encode(
        x=alt.X("key:N", axis=alt.Axis(title=None, labels=False, ticks=False))
    )

    # --- ticks/labels ---
    tick_dfs = [
        create_ticks_labels(df, norm_col, orig_col)
        for norm_col, orig_col in zip(normalized_columns, column_names)
    ]
    ticks_labels_df = pd.concat(tick_dfs, ignore_index=True)

    if show_ticks_labels:
        ticks = alt.Chart(ticks_labels_df).mark_tick(
            size=8, color="#ccc", orient="horizontal"
        ).encode(x="variable:N", y=alt.Y("value:Q", scale=alt.Scale(domain=y_domain)))

        labels = alt.Chart(ticks_labels_df).mark_text(
            align="center", baseline="middle", dx=-10
        ).encode(x="variable:N", y=alt.Y("value:Q", scale=alt.Scale(domain=y_domain)), text="label:N")
    else:
        ticks = alt.Chart(ticks_labels_df).mark_tick(
            size=8, opacity=0, orient="horizontal"
        ).encode(x="variable:N", y=alt.Y("value:Q", scale=alt.Scale(domain=y_domain)))

        labels = alt.Chart(ticks_labels_df).mark_text(
            align="center", baseline="middle", dx=-10, opacity=0
        ).encode(x="variable:N", y=alt.Y("value:Q", scale=alt.Scale(domain=y_domain)), text="label:N")

    # --- axis config ---
    axis_config_x = {
        "domain": False,
        "labelAngle": 0,
        "title": None,
        "tickColor": "#ccc" if show_ticks_labels else "transparent",
        "labelColor": "#000" if show_ticks_labels else "transparent",
        "grid": grid_on,
        "gridColor": "#ccc" if grid_on else "transparent",
    }
    axis_config_y = axis_config_x.copy()

    chart = (
        alt.layer(lines, rules, ticks, labels)
        .configure_axisX(**axis_config_x)
        .configure_axisY(**axis_config_y)
        .configure_view(stroke=None)
    )

    # --- background ---
    if background_value is not None:
        if isinstance(background_value, (tuple, list)) and len(background_value) == 3:
            r, g, b = background_value
            background_rgb = f"rgb({r},{g},{b})"
        else:
            v = int(background_value)
            background_rgb = f"rgb({v},{v},{v})"
        chart = chart.configure(background=background_rgb)

    # --- save ---
    if filename is not None:
        base_path, _ = os.path.splitext(filename)
        basename = os.path.basename(base_path)
        output_dir = os.path.dirname(filename)

        if svg_dir:
            os.makedirs(svg_dir, exist_ok=True)
            svg_filename = os.path.join(svg_dir, basename + ".svg")
        else:
            svg_filename = base_path + ".svg"

        png_filename = os.path.join(output_dir, basename + ".png")

        chart.save(svg_filename)
        if save_png:
            chart.save(png_filename, format="png", scale_factor=png_scale_factor)

    return chart

def redesign(combined_dir, output_dir, dominant_colors_json=None, category_colors_dir=None):
    combined_dir = os.fspath(combined_dir)
    output_dir = os.fspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    dominant_colors = None
    if dominant_colors_json is not None:
        try:
            with open(dominant_colors_json, "r") as f:
                dominant_colors = json.load(f)
            print(f"✅ Loaded dominant colors from: {dominant_colors_json}")
        except Exception as e:
            print(f"⚠️ Could not load dominant colors JSON: {e}")
            dominant_colors = None

    # find all combined CSVs in the directory
    csv_files = sorted(glob(os.path.join(combined_dir, "*_combined.csv")))
    if not csv_files:
        print(f"⚠️ No *_combined.csv files found in: {combined_dir}")
        return

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                print(f"⚠️ Skipping empty file: {csv_path}")
                continue
        except Exception as e:
            print(f"❌ Error reading {csv_path}: {e}")
            continue

        base = os.path.basename(csv_path)
        m = re.match(r"(.+)_combined\.csv$", base)
        image_id = m.group(1) if m else "Unknown"
        out_svg = os.path.join(output_dir, f"{image_id}_combined.svg")
        category_hsv_map = None

        if category_colors_dir is not None and image_id != "Unknown":
            json_path = find_image_json(image_id, category_colors_dir, recursive=True)
            if json_path is not None:
                category_colors_config = load_json_file(json_path)
                category_hsv_map = build_category_hsv_map_from_category_colors_json(df, category_colors_config)

        # Fallback to dominant colors if no per-image mapping found
        if category_hsv_map is None and dominant_colors is not None and image_id != "Unknown":
            category_hsv_map = build_category_hsv_map_from_dominant(df, image_id, dominant_colors)

        bg_gray = (215, 215, 215)
        generate_plot(df, filename=out_svg, category_hsv_map=category_hsv_map)
                    # background_value=bg_gray)

        # generate_plot_new(df, filename=out_svg, category_hsv_map=category_hsv_map, grid_on=True,
        #                   width=470, height=470, background_value=bg_gray)

        print(f"✅ Plot saved: {out_svg}")

def redesign_old(combined_dir, output_dir):
    combined_dir = os.fspath(combined_dir)
    output_dir = os.fspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # find all combined CSVs in the directory
    csv_files = sorted(glob(os.path.join(combined_dir, "*_combined.csv")))
    if not csv_files:
        print(f"⚠️ No *_combined.csv files found in: {combined_dir}")
        return

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                print(f"⚠️ Skipping empty file: {csv_path}")
                continue
        except Exception as e:
            print(f"❌ Error reading {csv_path}: {e}")
            continue

        # image id from the filename (not the full path)
        base = os.path.basename(csv_path)
        m = re.match(r"(.+)_combined\.csv$", base)
        image_id = m.group(1) if m else "Unknown"

        out_svg = os.path.join(output_dir, f"{image_id}_combined.svg")
        # generate_plot will also make a PNG if you pass save_png=True

        # with open("category_colors.json", "r") as f:
        #     config = json.load(f)
        #
        # category_hsv_map = config["category_colors"]

        # category_hsv_map = {
        #     "08Qf": {"h": 0.87, "s": 1, "v": 1},
        #     "UdMbD": {"h": 0.37, "s": 1, "v": 1},
        # }
        # category_hsv_map_171 = {
        #     "cat1": {"h": 0.08, "s": 1, "v": 1},
        #     "cat2": {"h": 0.33, "s": 1, "v": 1},
        #       "cat3": {"h": 0.57, "s": 1, "v": 1},
        # }
        category_hsv_map_261 = {
            "cat1": {"h": 0.48, "s": 1, "v": 1},
            "cat2": {"h": 0.58, "s": 1, "v": 1},
            "cat3": {"h": 0.8, "s": 1, "v": 1},
        }
        category_hsv_map_15 = {
            "0YKdjv": {
            "h": 0.83,
            "s": 1,
            "v": 1
        },
        "JBvaS": {
            "h": 0.53,
            "s": 1,
            "v": 1
        },
        "MuWL9a": {
            "h": 0.3,
            "s": 1,
            "v": 1
        }
        }
        generate_plot(df, filename=out_svg, category_hsv_map=category_hsv_map_15)
        #generate_plot(df, filename=out_svg)
        print(f"✅ Plot saved: {out_svg}")


# -----------------------------

# Run
# -----------------------------

#if __name__ == "__main__":
    #main("/home/poonam/myworkspace/PaCoNet/outputs/syns/redesigned/1")

    #main("/home/poonam/myworkspace/PaCoNet/data/real_plots/2025-11-18_16-17-52/dhlp_output")
    #redesign("/home/poonam/myworkspace/PaCoNet/outputs/syns/redesigned/1", "/home/poonam/myworkspace/PaCoNet/outputs/syns/redesigned/1_plots")
    #redesign("/home/poonam/myworkspace/PaCoNet/data/real_plots/2025-11-18_16-17-52/dhlp_output",
             #"/home/poonam/myworkspace/PaCoNet/data/real_plots/2025-11-18_16-17-52/dhlp_output/redesigned/s15")


# stitch keep all was used for synth
# stitch segment was used for real