import os
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


def stitch_parallel_coordinates(file_paths, column_map, category_id=1, threshold=5.0):
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

        stitched_df = stitch_parallel_coordinates(file_paths, column_map, category_id=category, threshold=threshold)
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

        stitched_df = stitch_parallel_coordinates(file_paths, column_map, category_id=label, threshold=threshold)
        stitched_path = base_dir / f"stitched_{label}.csv"
        stitched_df.to_csv(stitched_path, index=False)
        print(f"✅ Saved: {stitched_path}")
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

    lines = base.mark_line(opacity=1).encode(
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


def redesign(combined_dir, output_dir, dominant_colors_json=None, category_colors_dir=None):
    combined_dir = os.fspath(combined_dir)
    output_dir = os.fspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load dominant colors once (if provided)
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

        # image id from the filename (not the full path)
        base = os.path.basename(csv_path)
        m = re.match(r"(.+)_combined\.csv$", base)
        image_id = m.group(1) if m else "Unknown"

        out_svg = os.path.join(output_dir, f"{image_id}_combined.svg")

        # Build category_hsv_map from dominant_colors.json (if available)
        # Option A: prefer explicit category_colors JSON (like image_2.json)
        category_hsv_map = None

        # Prefer per-image jsons from folder: image_<id>.json
        if category_colors_dir is not None and image_id != "Unknown":
            json_path = find_image_json(image_id, category_colors_dir, recursive=True)
            if json_path is not None:
                category_colors_config = load_json_file(json_path)
                category_hsv_map = build_category_hsv_map_from_category_colors_json(df, category_colors_config)

        # Fallback to dominant colors if no per-image mapping found
        if category_hsv_map is None and dominant_colors is not None and image_id != "Unknown":
            category_hsv_map = build_category_hsv_map_from_dominant(df, image_id, dominant_colors)

        generate_plot(df, filename=out_svg, category_hsv_map=category_hsv_map)

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
