import os
import re
import json
from pathlib import Path
from glob import glob
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


# -----------------------------
# Helper Functions
# -----------------------------

def extract_image_id(filename):
    """Extracts the image ID from a filename."""
    match = re.match(r"(\d+)_crop_", filename)
    return match.group(1) if match else None


def extract_crop_number(path):
    """Extracts the crop number from a file path."""
    match = re.search(r'crop_(\d+)_cat', path)
    return int(match.group(1)) if match else -1


def extract_category(path):
    """Extracts the category ID from a file path."""
    match = re.search(r'cat_(\d+)', path)
    return int(match.group(1)) if match else -1


# -----------------------------
# Step 1: JSON to CSV Conversion (MODIFIED FOR COLUMN ORDER)
# -----------------------------

def convert_json_to_csv(json_dir):
    """
    Converts JSON files containing line segments into CSV files.
    Each line segment's start (x1, y1) and end (x2, y2) coordinates,
    along with crop number and category, are saved.
    """
    pattern = re.compile(r'crop_(\d+)_cat_(\d+)')
    json_dir = Path(json_dir)

    print(f"🔄 Converting JSON files in: {json_dir}")
    json_files_found = False
    for json_file in json_dir.glob("*.json"):
        json_files_found = True
        print(f"   Attempting to load: {json_file.name}")
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Error decoding JSON from {json_file.name}: {e}")
            print(f"   Please check if {json_file.name} is a valid JSON file.")
            continue

        lines = data.get("lines", [])
        match = pattern.search(json_file.stem)

        if match:
            crop_num = int(match.group(1))
            cat_id = int(match.group(2))
        else:
            print(f"⚠️ Warning: Filename '{json_file.name}' does not match expected pattern. Skipping.")
            continue

        rows = []
        for p1, p2 in lines:
            x1, y1 = p1
            x2, y2 = p2

            row = {
                "crop_num": crop_num,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cat": cat_id  # Moved 'cat' to the end of the dictionary
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = json_file.with_name(f"{json_file.stem}_xy.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ JSON converted to XY CSV: {csv_path}")

    if not json_files_found:
        print(f"⚠️ No JSON files found in {json_dir}. Please ensure your JSONs are in this directory.")


# -----------------------------
# Step 2: Stitching for XY Plotting (CORRECTED LOGIC)
# -----------------------------

def stitch_xy_coordinates(base_dir, image_id, crop_width=224.0, threshold_distance=10.0):
    """
    Stitches line segments from multiple cropped images into continuous lines
    for XY plotting, handling stitching independently for each category.

    Args:
        base_dir (Path): The base directory where the _xy.csv files are located.
        image_id (str): The ID of the image being processed (e.g., "92").
        crop_width (float): The assumed width of each individual crop in pixels.
                            This is crucial for calculating X-offsets.
        threshold_distance (float): The maximum distance between a line's end
                                    point and another line's start point to be
                                    considered for stitching.
    Returns:
        pd.DataFrame: A DataFrame containing the stitched line segments,
                      with global x and y coordinates. Each row represents
                      a segment of a potentially longer stitched line.
    """
    base_dir = Path(base_dir)
    print(f"\n🔗 Stitching XY coordinates for image ID: {image_id}")

    # Load all relevant _xy.csv files into a single DataFrame first
    all_dfs = []
    # Get all unique crop numbers and sort them to ensure correct processing order
    # This is important because glob might not return files in numerical order
    all_xy_csv_files = sorted(
        base_dir.glob(f"{image_id}_crop_*_cat_*_xy.csv"),
        key=lambda f: extract_crop_number(str(f))
    )

    if not all_xy_csv_files:
        print(f"⚠️ No _xy.csv files found for image ID {image_id} in {base_dir}. Run JSON conversion first.")
        return pd.DataFrame()

    # Read all CSVs into a list of DataFrames
    for file_path in all_xy_csv_files:
        df = pd.read_csv(file_path)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print(f"⚠️ All CSV files for image ID {image_id} were empty after conversion.")
        return pd.DataFrame()

    # Concatenate all data into one DataFrame for easier filtering
    combined_raw_df = pd.concat(all_dfs, ignore_index=True)

    # Get all unique categories present in the data
    unique_categories = sorted(combined_raw_df['cat'].unique())

    # Get all unique crop numbers in order
    unique_crop_nums = sorted(combined_raw_df['crop_num'].unique())

    final_stitched_data = []  # This will hold all stitched segments from all categories

    for cat_id in unique_categories:
        print(f"\n   Processing Category {cat_id}...")

        # This list will store dictionaries for the current category's stitched lines.
        # Each stitched line will have a 'cat' (category) and 'points' (a list of [x,y] tuples).
        stitched_lines_for_category = []
        x_offset_for_category = 0.0  # Reset offset for each new category

        for i, crop_num in enumerate(unique_crop_nums):
            # Filter data for the current category and current crop
            df_current_crop_cat = combined_raw_df[
                (combined_raw_df['crop_num'] == crop_num) &
                (combined_raw_df['cat'] == cat_id)
                ].copy()  # Use .copy() to avoid SettingWithCopyWarning

            if df_current_crop_cat.empty:
                # If this crop doesn't have data for this category, still increment offset
                # to maintain correct global X for subsequent crops of this category.
                x_offset_for_category += crop_width
                continue  # Skip to next crop_num for this category

            print(f"      - Crop {crop_num} (Cat {cat_id}) with current x_offset: {x_offset_for_category}")

            # Apply the x-offset to the current crop's lines for this category
            df_current_crop_cat['x1_global'] = df_current_crop_cat['x1'] + x_offset_for_category
            df_current_crop_cat['x2_global'] = df_current_crop_cat['x2'] + x_offset_for_category

            current_crop_segments = []
            for _, row in df_current_crop_cat.iterrows():
                current_crop_segments.append({
                    "cat": row["cat"],
                    "start_point": np.array([row["x1_global"], row["y1"]]),
                    "end_point": np.array([row["x2_global"], row["y2"]])
                })

            if i == 0:  # This is the first crop for this specific category
                for segment in current_crop_segments:
                    stitched_lines_for_category.append({
                        "cat": segment["cat"],
                        "points": [list(segment["start_point"]), list(segment["end_point"])]
                    })
            else:
                # For subsequent crops within this category, try to stitch to existing lines
                current_segments_stitched_flags = [False] * len(current_crop_segments)

                for stitched_line_idx, stitched_line in enumerate(stitched_lines_for_category):
                    last_point_of_stitched_line = stitched_line["points"][-1]
                    # No need to check category here, as we are already processing within a single category

                    best_match_segment_idx = -1
                    min_dist = float('inf')

                    # Find the closest starting point in the current crop for the same category
                    for current_segment_idx, current_segment in enumerate(current_crop_segments):
                        if current_segments_stitched_flags[current_segment_idx]:
                            continue  # Already stitched this segment

                        # We only consider segments from the current category, which is already filtered
                        dist = np.linalg.norm(last_point_of_stitched_line - current_segment["start_point"])
                        if dist < min_dist:
                            min_dist = dist
                            best_match_segment_idx = current_segment_idx

                    # If a close enough match is found, extend the stitched line
                    if best_match_segment_idx != -1 and min_dist <= threshold_distance:
                        matched_segment = current_crop_segments[best_match_segment_idx]
                        stitched_line["points"].append(list(matched_segment["end_point"]))
                        current_segments_stitched_flags[best_match_segment_idx] = True

                # Add any remaining (unstitched) segments from the current crop as new lines
                for current_segment_idx, current_segment in enumerate(current_crop_segments):
                    if not current_segments_stitched_flags[current_segment_idx]:
                        stitched_lines_for_category.append({
                            "cat": current_segment["cat"],
                            "points": [list(current_segment["start_point"]), list(current_segment["end_point"])]
                        })

            x_offset_for_category += crop_width  # Update offset for the next crop within this category

        # After processing all crops for this category, add its stitched lines to the final list
        for line in stitched_lines_for_category:
            for j in range(len(line["points"]) - 1):
                p1 = line["points"][j]
                p2 = line["points"][j + 1]
                final_stitched_data.append({
                    "cat": int(line["cat"]),
                    "x1_stitched": p1[0],
                    "y1_stitched": p1[1],
                    "x2_stitched": p2[0],
                    "y2_stitched": p2[1]

                })

    stitched_df = pd.DataFrame(final_stitched_data)
    output_path = base_dir / f"{image_id}_stitched_xy.csv"
    stitched_df.to_csv(output_path, index=False)
    print(f"✅ Stitched XY CSV saved: {output_path}")
    return stitched_df


# -----------------------------
# Step 3: Plotting
# -----------------------------

def plot_stitched_xy(df, title="Stitched Lines across Crops"):
    """
    Plots the stitched line segments using matplotlib.

    Args:
        df (pd.DataFrame): DataFrame containing columns 'x1_stitched', 'y1_stitched',
                           'x2_stitched', 'y2_stitched', and 'cat'.
        title (str): The title for the plot.
    """
    if df.empty:
        print("No data to plot.")
        return

    plt.figure(figsize=(18, 8))  # Adjust figure size for better visualization

    # Get unique categories for coloring
    unique_categories = df['cat'].unique()
    colors = plt.cm.get_cmap('tab10', len(unique_categories))  # Use a colormap for distinct colors

    # Keep track of labels already added to the legend
    added_labels = set()

    for i, cat_id in enumerate(unique_categories):
        cat_df = df[df['cat'] == cat_id]

        # Determine the label for this category
        current_label = f'Category {cat_id}'

        # Check if this category label has already been added to the legend
        label_to_use = current_label if current_label not in added_labels else ""

        for _, row in cat_df.iterrows():
            plt.plot([row['x1_stitched'], row['x2_stitched']],
                     [row['y1_stitched'], row['y2_stitched']],
                     color=colors(i),
                     linewidth=2,
                     label=label_to_use)  # Use the determined label

            # After plotting the first segment for this category, clear the label
            # so subsequent segments of the same category don't create duplicate legend entries.
            if label_to_use:
                added_labels.add(current_label)
                label_to_use = ""  # Clear label for subsequent segments of the same category

    plt.xlabel("Global X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.title(title)
    plt.legend(title="Categories")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_aspect('equal', adjustable='box')  # Maintain aspect ratio
    plt.tight_layout()  # Adjust layout to prevent labels overlapping
    plt.show()


# -----------------------------
# Main Execution Pipeline
# -----------------------------

# def main():
#     """
#     Main function to run the entire pipeline:
#     1. Convert JSONs to XY CSVs.
#     2. Stitch the XY coordinates across crops.
#     3. Plot the stitched lines.
#     """
#     # Define the base directory where your JSON files are located
#     # If your JSONs are in the same directory as this script, use Path(".").resolve()
#     # Otherwise, specify the full path, e.g., Path("/path/to/your/json_files")
#     base_directory = Path("../outputs/reals/redesigned").resolve()
#
#     # Define the image ID you want to process (e.g., "92" from "92_crop_6_cat_1.json")
#     image_id_to_process = "92"
#
#     # Define the assumed width of each crop.
#     # This is crucial. If your crops have different widths, you'll need to
#     # adjust the `stitch_xy_coordinates` function to take a list of widths
#     # or infer them from image metadata.
#     # For the provided JSONs, the x-coordinates range up to ~224, so 224 is a reasonable guess.
#     assumed_crop_width = 224.0
#
#     # Define the threshold for stitching.
#     # This is the maximum distance between an end point of a line in one crop
#     # and a start point of a line in the next crop for them to be stitched.
#     # Adjust this value based on how much misalignment your model produces.
#     stitching_threshold = 10.0  # Example: 10 pixels
#
#     print("🚀 Starting line stitching and plotting pipeline...")
#
#     # Step 1: Convert JSONs to XY CSVs
#     convert_json_to_csv(base_directory)
#
#     # Step 2: Stitch the XY coordinates
#     stitched_df = stitch_xy_coordinates(base_directory, image_id_to_process,
#                                         crop_width=assumed_crop_width,
#                                         threshold_distance=stitching_threshold)
#
#     # Step 3: Plot the stitched lines
#     if not stitched_df.empty:
#         plot_stitched_xy(stitched_df,
#                          title=f"Stitched Lines for Image {image_id_to_process} (Threshold: {stitching_threshold})")
#     else:
#         print("No stitched data to plot.")
#
#
# # Entry point for the script
# if __name__ == "__main__":
#     main()