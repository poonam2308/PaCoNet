
import os
import pandas as pd
import altair as alt
import numpy as np
import json
import colorsys

# Convert HSV to RGB for Altair compatibility
def hsv_to_rgb(h, s, v):
    return tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

# Normalize a column to a [0,1] range
def normalize_column(df, col_name):
    min_val = df[col_name].min()
    max_val = df[col_name].max()
    norm_col_name = f" {col_name}"
    df[norm_col_name] = 1 - (df[col_name] - min_val) / (max_val - min_val)
    return norm_col_name

def generate_plots(df, filename, custom_hsv_colors=None, width=600, height=300):
    column_names = list(df.columns)[:-1]
    color_column = df.columns[-1]

    normalized_columns = [normalize_column(df, col) for col in column_names]
    unique_categories = df[color_column].unique()

    if custom_hsv_colors:
        category_colors = {
            category: hsv_to_rgb(*(h / 360, s / 100, v / 100))
            for category, (h, s, v) in zip(unique_categories, custom_hsv_colors)
        }
    else:
        category_colors = {
            category: hsv_to_rgb(round(h / len(unique_categories), 2), 1, 1)
            for h, category in enumerate(unique_categories)
        }

    base = alt.Chart(df).transform_window(index="count()").transform_fold(normalized_columns).transform_calculate(
        mid="(datum.value + datum.value) / 2"
    ).properties(width=width, height=height)

    lines = base.mark_line(opacity=0.3).encode(
        x='key:N',
        y=alt.Y('value:Q', axis=None),
        color=alt.Color(f"{color_column}:N", scale=alt.Scale(
            domain=list(category_colors.keys()),
            range=['rgb({},{},{})'.format(*color) for color in category_colors.values()]
        ), legend=None),
        detail="index:N",
        tooltip=normalized_columns
    )

    rules = base.mark_rule(color="#ccc", tooltip=None).encode(
        x="key:N",
        detail="count():Q"
    )

    chart = alt.layer(lines, rules).configure_axisX(
        domain=False, labelAngle=0, labelColor="transparent", tickColor="transparent", title=None
    ).configure_view(stroke=None)

    chart.save(filename)
    return normalized_columns

# Process all CSV files in a directory
def generate_allcat_hsvplots_for_directory(input_dir, output_dir, color_json_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(color_json_path, "r") as f:
        raw_color_data = json.load(f)

        # Convert from list of dicts → {filename: [h, s, v]}
        image_hsv_map = {}
        for entry in raw_color_data:
            fname = entry.get("filename", "").lower()
            hsv = entry.get("color_hsv")
            if fname and hsv:
                image_hsv_map[fname] = [[hsv['h'], hsv['s'], hsv['v']]]

    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    for csv_filename in csv_files:
        file_path = os.path.join(input_dir, csv_filename)
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"Skipping empty file: {csv_filename}")
                continue
        except Exception as e:
            print(f"Error reading {csv_filename}: {e}")
            continue

        color_column = df.columns[-1]
        unique_categories = df[color_column].unique()

        selected_categories = unique_categories
        df = df[df[color_column].isin(selected_categories)]
        if df.empty:
            print(f"No matching categories in file: {csv_filename}")
            continue

        # Match .csv name to .png image key in JSON
        image_base = os.path.splitext(csv_filename)[0].lower()
        image_key = image_base + ".png"
        hsv_list = image_hsv_map.get(image_key, None)
        if hsv_list is None:
            print(f"No HSV colors found for: {image_key}, falling back to default.")

        output_filename = os.path.join(output_dir, f"{image_base}.svg")
        generate_plots(df, output_filename, custom_hsv_colors=hsv_list)

    print(f"All plots generated and saved in {output_dir}.")

# Example usage
# if __name__ == "__main__":
#     input_dir = 'predicted_data/mcat/color_files'
#     output_dir = 'predicted_data/mcat_plots/color_filee_plots'
#     dominant_color_files = "../dataset/real_world_test_data/final_set_raw_images/dominant_colors.json"
#     generate_allcat_hsvplots_for_directory(input_dir, output_dir, dominant_color_files)

