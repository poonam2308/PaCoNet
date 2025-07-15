import os
import re
import json
from pathlib import Path
from glob import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cdist
# -----------------------------
# Helper Functions
# -----------------------------

def extract_image_id(filename):
    match = re.match(r"(\d+)_crop_", filename)
    return match.group(1) if match else None

def extract_crop_number(path):
    match = re.search(r'crop_(\d+)_cat', path)
    return int(match.group(1)) if match else -1

def extract_category(path):
    match = re.search(r'cat_(\d+)', path)
    return match.group(0) if match else None

def get_latest_path(base_path):
    updated_path = base_path.replace(".csv", "_updated.csv")
    return updated_path if os.path.exists(updated_path) else base_path

# -----------------------------
# Step 1: JSON to CSV Conversion
# -----------------------------

def convert_json_to_csv(json_dir):
    pattern = re.compile(r'crop_(\d+)_cat_(\d+)')
    json_dir = Path(json_dir)

    for json_file in json_dir.glob("*.json"):
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        lines = data.get("lines", [])
        match = pattern.search(json_file.stem)
        crop = match.group(1) if match else "unknown"
        cat = match.group(2) if match else "unknown"

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
                "cat": cat
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = json_file.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ JSON converted to CSV: {csv_path}")


def get_crop_csvs_and_column_map(base_dir, image_id, category):
    base_dir = Path(base_dir)
    pattern = rf"{image_id}_crop_(\d+)_cat_{category}_0\.csv"

    matched_files = [
        f for f in base_dir.glob(f"{image_id}_crop_*_cat_{category}_0.csv")
        if re.match(pattern, f.name)
    ]

    sorted_files = sorted(
        matched_files,
        key=lambda f: int(re.match(pattern, f.name).group(1))
    )

    file_paths = [str(f) for f in sorted_files]
    column_map = []

    for f in sorted_files:
        crop_num = re.match(pattern, f.name).group(1)
        column_map.append((f"crop_{crop_num}_y1", f"crop_{crop_num}_y2"))

    return file_paths, column_map


def stitch_parallel_coordinates(file_paths, column_map, category_id=1, threshold=5.0):
    dfs = [
        pd.read_csv(fp).rename(columns={y1: "y1", y2: "y2"})
        for fp, (y1, y2) in zip(file_paths, column_map)
    ]

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
    base_dir = Path(base_dir)
    pattern = rf"{image_id}_crop_(\d+)_cat_{category}\.csv"

    matched_files = [
        f for f in base_dir.glob(f"{image_id}_crop_*_cat_{category}.csv")
        if re.match(pattern, f.name)
    ]

    sorted_files = sorted(
        matched_files,
        key=lambda f: int(re.match(pattern, f.name).group(1))
    )

    file_paths = [str(f) for f in sorted_files]
    column_map = []

    for f in sorted_files:
        crop_num = re.match(pattern, f.name).group(1)
        column_map.append((f"crop_{crop_num}_y1", f"crop_{crop_num}_y2"))

    return file_paths, column_map


def stitch_parallel_coordinates(file_paths, column_map, category_id=1, threshold=5.0):
    dfs = [
        pd.read_csv(fp).rename(columns={y1: "y1", y2: "y2"})
        for fp, (y1, y2) in zip(file_paths, column_map)
    ]

    paths = [
        {"axis1": row["y1"], "axis2": row["y2"]}
        for _, row in dfs[0].iterrows()
    ]

    for i in range(1, len(dfs)):
        new_paths = []
        for path in paths:
            last_axis_val = path[f"axis{i + 1}"]
            candidates = dfs[i]
            closest = candidates.iloc[(candidates["y1"] - last_axis_val).abs().argsort()[:1]]
            if abs(closest["y1"].values[0] - last_axis_val) <= threshold:
                extended_path = path.copy()
                extended_path[f"axis{i + 2}"] = closest["y2"].values[0]
                new_paths.append(extended_path)
        paths = new_paths

    for path in paths:
        path["cat"] = category_id

    return pd.DataFrame(paths)


def process_all_categories_and_combine(base_dir, image_id, threshold=10.0):
    base_dir = Path(base_dir)

    # Detect all categories available for the image
    all_crop_files = list(base_dir.glob(f"{image_id}_crop_*_cat_*.csv"))
    category_ids = sorted(set(
        int(re.search(rf"{image_id}_crop_\d+_cat_(\d+)\.csv", f.name).group(1))
        for f in all_crop_files
    ))

    combined_rows = []

    for category in category_ids:
        print(f"🔄 Processing category {category}...")
        file_paths, column_map = get_crop_csvs_and_column_map(base_dir, image_id, category)
        if not file_paths:
            print(f"⚠️ No files found for category {category}")
            continue

        stitched_df = stitch_parallel_coordinates(file_paths, column_map, category_id=category, threshold=threshold)
        stitched_path = base_dir / f"stitched_category{category}.csv"
        stitched_df.to_csv(stitched_path, index=False)
        print(f"✅ Saved: {stitched_path}")

        combined_rows.append(stitched_df)

    # Combine all stitched results
    if combined_rows:
        final_df = pd.concat(combined_rows, ignore_index=True)
        output_path = base_dir / f"{image_id}_combined.csv"
        final_df.to_csv(output_path, index=False)
        print(f"📦 Final combined saved: {output_path}")
    else:
        print("⚠️ No categories stitched. Nothing to combine.")

# 🧪 Example usage:
# process_all_categories_and_combine(base_dir="data/crops", image_id="171", threshold=10.0)

def main(base_dir="29_05/29"):
    convert_json_to_csv(base_dir)

    csv_files = glob(os.path.join(base_dir, "*_crop_*_cat_*.csv"))
    image_ids = sorted(set(
        re.match(r"(\d+)_crop_", os.path.basename(f)).group(1)
        for f in csv_files if re.match(r"(\d+)_crop_", os.path.basename(f))
    ))

    for image_id in image_ids:
        print(f"\n🚀 Starting pipeline for image ID: {image_id}")
        # process_and_chain_csvs(base_dir, image_id)
        combine_chained_csvs(base_dir, image_id)

# # -----------------------------
# # Run
# # -----------------------------
#
# if __name__ == "__main__":
#     main("data/sd_data_redesign/output/r_171_1_")  # Replace this with your folder path if different
