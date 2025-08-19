import os
import pandas as pd
import altair as alt
import numpy as np
import json
import colorsys
import re
from pathlib import Path
from glob import glob


# Convert HSV to RGB for Altair compatibility
def hsv_to_rgb(h, s, v):
    """Converts HSV color values (0-360, 0-100, 0-100) to RGB (0-255)."""
    return tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h / 360, s / 100, v / 100))


# --- XY Plotting with Optional HSV Colors ---
def generate_xy_plot(df, filename, width=800, height=600, crop_width=224.0, custom_hsv_colors=None):
    """
    Generates an XY plot for stitched line segments using Altair, with optional HSV-based coloring.
    """
    if df.empty:
        print(f"Skipping empty DataFrame for XY plot: {filename}")
        return

    df['segment_id'] = range(len(df))
    unique_categories = sorted(df['cat'].unique())

    # Apply HSV category mapping
    category_colors = {}
    if custom_hsv_colors:
        for cat in unique_categories:
            if str(cat) in custom_hsv_colors:
                h, s, v = custom_hsv_colors[str(cat)]
                # Scale h, s, v from [0–1] to [0–360], [0–100], [0–100]
                category_colors[cat] = hsv_to_rgb(h * 360, s * 100, v * 100)

            else:
                # Use default HSV with non-zero value for saturation and brightness
                default_h = round((int(cat) / len(unique_categories)) * 360, 2)
                category_colors[cat] = hsv_to_rgb(default_h, 100, 100)

    else:
        for i, cat in enumerate(unique_categories):
            category_colors[cat] = hsv_to_rgb(round((i / len(unique_categories)) * 360, 2), 100, 100)

    # FIX: sort keys to preserve correct mapping
    # sorted_keys = sorted(category_colors.keys(), key=lambda x: int(x))
    # altair_color_domain = sorted_keys
    # altair_color_range = ['rgb({},{},{})'.format(*category_colors[k]) for k in sorted_keys]

    altair_color_domain = [cat for cat in unique_categories]  # Use sorted unique categories directly
    altair_color_range = ['rgb({},{},{})'.format(*category_colors[cat]) for cat in unique_categories]

    # Vertical reference lines
    max_x = df['x2_stitched'].max() if not df.empty else 0
    vertical_line_x_coords = np.arange(0, max_x + crop_width, crop_width)
    vertical_lines_df = pd.DataFrame({'x_pos': vertical_line_x_coords})

    line_chart = alt.Chart(df).mark_line().encode(
        # x=alt.X('x1_stitched:Q', title='Global X-coordinate', axis=alt.Axis(grid=False)),
        # y=alt.Y('y1_stitched:Q', title='Y-coordinate', axis=alt.Axis(grid=False)),
        x=alt.X('x1_stitched:Q', axis=None),
        y=alt.Y('y1_stitched:Q', axis=None, scale=alt.Scale(reverse=True)),

        x2='x2_stitched:Q',
        y2='y2_stitched:Q',
        color=alt.Color('cat:N', scale=alt.Scale(domain=altair_color_domain,
                                                 range=altair_color_range), legend=None),
        detail='segment_id:N',
        tooltip=['x1_stitched', 'y1_stitched', 'x2_stitched', 'y2_stitched', 'cat']
    )

    vertical_rules = alt.Chart(vertical_lines_df).mark_rule(
        color='lightgray', strokeWidth=1, strokeDash=[2, 2]
    ).encode(x='x_pos:Q')

    chart = alt.layer(line_chart, vertical_rules).properties(
        width=width,
        height=height
    ).interactive()

    chart.save(filename)
    print(f"✅ Generated stitched XY plot: {filename}")


# --- Main Driver for Processing Stitched CSVs ---
def process_stitched_xy_csvs(input_dir, output_dir, crop_width_val, hsv_json_path=None):
    """
    Processes all *_stitched_xy.csv files and generates stitched XY SVG plots with optional HSV colors.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    stitched_csv_files = [f for f in os.listdir(input_dir) if f.endswith('_stitched_xy.csv')]
    if not stitched_csv_files:
        print(f"⚠️ No *_stitched_xy.csv files found in {input_dir}.")
        return

    from collections import defaultdict

    # Load and average hue per category across all crops
    hue_accumulator = defaultdict(list)
    if hsv_json_path and os.path.exists(hsv_json_path):
        try:
            with open(hsv_json_path, "r") as f:
                entries = json.load(f)
                custom_hsv_colors = {}

                # for entry in entries:
                #     match = re.search(r'_cat_(\d+)', entry["filename"])
                #     if match:
                #         cat_id = match.group(1)
                #         if cat_id not in custom_hsv_colors:
                #             hue = float(entry["hue"])
                #             custom_hsv_colors[cat_id] = [hue * 360, 100, 100]  # Convert to degrees

                for entry in entries:
                    fname = entry.get("filename", "").lower()
                    hsv = entry.get("color_hsv")
                    if fname and hsv:
                        h, s, v = hsv['h'], hsv['s'], hsv['v']
                        # You can extract cat ID from filename if needed
                        match = re.search(r'_cat_(\d+)', fname)
                        if match:
                            cat_id = match.group(1)
                            custom_hsv_colors[cat_id] = [h, s, v]

            print(f"✅ Loaded HSV for categories: {custom_hsv_colors}")
        except Exception as e:
            print(f"❌ Failed to load HSV from hue_summary.json: {e}")
            custom_hsv_colors = {}

    print(f"\n📊 Processing stitched XY CSVs in: {input_dir}")
    for csv_filename in stitched_csv_files:
        file_path = os.path.join(input_dir, csv_filename)
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"⚠️ Skipping empty file: {csv_filename}")
                continue
        except Exception as e:
            print(f"❌ Error reading {csv_filename}: {e}")
            continue

        match = re.match(r"(\d+)_stitched_xy\.csv", csv_filename)
        image_id = match.group(1) if match else "Unknown"
        output_svg = os.path.join(output_dir, f"{image_id}_stitched_xy_plot.svg")

        generate_xy_plot(df, output_svg, crop_width=crop_width_val, custom_hsv_colors=custom_hsv_colors)

    print(f"\n✅ All stitched XY plots saved in {output_dir}")

