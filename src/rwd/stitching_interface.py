# stitching_interface.py

from src.rwd.demo_stitch_xy import extract_image_id, convert_json_to_csv, process_all_categories_and_combine
from pathlib import Path
import pandas as pd

def run_stitching_from_json_dir(json_dir, threshold=10.0):
    base_dir = Path(json_dir)
    convert_json_to_csv(base_dir)

    # Get image ID from one of the JSON files
    json_files = list(base_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError("No JSON files found in the given directory.")

    image_ids = set(filter(None, [extract_image_id(f.name) for f in json_files]))
    if not image_ids:
        raise ValueError("Could not extract any image_id from the JSON filenames.")
    image_id = sorted(image_ids)[0]

    process_all_categories_and_combine(base_dir=base_dir, image_id=image_id, threshold=threshold)

    # Read the outputs for Gradio display
    stitched_csvs = sorted(base_dir.glob("stitched_category*.csv"))
    combined_csv = base_dir / f"{image_id}_combined.csv"

    dfs = [(f.name, pd.read_csv(f)) for f in stitched_csvs]
    combined_df = pd.read_csv(combined_csv) if combined_csv.exists() else pd.DataFrame()

    return dfs, ("All Categories", combined_df)
