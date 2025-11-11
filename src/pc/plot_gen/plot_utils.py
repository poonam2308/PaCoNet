import math

import pandas as pd
import numpy as np
import colorsys
import re
import json
import os

def generate_large_color_palette(n=100):
    return [tuple(int(x * 255) for x in colorsys.hsv_to_rgb(i / n, 1, 1)) for i in range(n)]


def generate_hsv_pool(n):
    return [{'h': round(i / n, 2), 's': 1, 'v': 1} for i in range(n)]

# Convert HSV to RGB for Altair compatibility
def hsv_to_rgb(h, s, v):
    return tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

# Generate tick labels for axis
def create_ticks_labels(df, col_name, orig_col_name):
    dfcol = df[col_name]
    if dfcol.dtype == object:
        dfcol = dfcol.astype('category')
    if isinstance(dfcol.dtype, pd.CategoricalDtype):
        xlabs = dfcol.cat.categories.values
        xnorm = np.linspace(0, 1, len(dfcol.cat.categories))
    else:
        nticks = 8
        xnorm = np.linspace(0, 1, nticks)
        xorig = np.linspace(df[orig_col_name].min(), df[orig_col_name].max(), nticks)
        xlabs = [f'{x:.1f}' for x in xorig]
    return pd.DataFrame({'value': xnorm, 'label': xlabs, 'variable': col_name})

# Normalize column values to the range [0, 1]
def normalize_column(df, col_name):
    min_val = df[col_name].min()
    max_val = df[col_name].max()
    norm_col_name = f" {col_name}"
    df[norm_col_name] = (df[col_name] - min_val) / (max_val - min_val)
    return norm_col_name

def normalize_column_reverse(df, col_name):
    min_val = df[col_name].min()
    max_val = df[col_name].max()
    norm_col_name = f" {col_name}"
    df[norm_col_name] = 1-((df[col_name] - min_val) / (max_val - min_val))
    return norm_col_name

def calculate_pixel_positions(df, normalized_columns, height):
    pixel_positions = {}
    for col in normalized_columns:
        col_name = col.strip()
        pixel_positions[col_name] = height - (df[col] * height)
    return pixel_positions

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

def split_data(input_file, train_file, valid_file, train_ratio=0.8):
    with open(input_file, 'r') as f:
        all_data = json.load(f)
    # Group data by images (e.g., image_1, image_2)
    grouped_data = {}
    for item in all_data:
        image_key = item['filename'].split('_crop')[0]
        if image_key not in grouped_data:
            grouped_data[image_key] = []
        grouped_data[image_key].append(item)

    grouped_list = list(grouped_data.values())
    np.random.shuffle(grouped_list)

    train_size = int(len(grouped_list) * train_ratio)
    train_data = [item for group in grouped_list[:train_size] for item in group]
    valid_data = [item for group in grouped_list[train_size:] for item in group]

    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=4)

    with open(valid_file, 'w') as f:
        json.dump(valid_data, f, indent=4)

def safe_join(base_dir, file_path):
    if not os.path.isabs(file_path) and not os.path.dirname(file_path):
        return os.path.join(base_dir, file_path)
    return file_path

def round_half_up(x):
    """Round halves up (0.5 -> 1, -0.5 -> 0) for consistent pixel placement."""
    return int(math.floor(float(x) + 0.5))


def update_lines(json_file, output_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    new_data = []
    new_height = 224
    original_height = 321
    scale_y = new_height / original_height  # Fixed scaling factor for height

    for item in data:
        filename = item["filename"]
        updated_lines = []

        for line in item["lines"]:
            original_width = line[2]  # Extract original width from the third element of lines
            scale_x = new_height / original_width  # Scaling factor for width

            # Apply scaling
            new_line = [
                round(line[0] * scale_x,2),
                round(line[1] * scale_y, 2),
                round(line[2] * scale_x, 2),
                round(line[3] * scale_y, 2)
            ]
            updated_lines.append(new_line)

        new_data.append({
            "filename": filename,
            "lines": updated_lines
        })

    with open(output_file, 'w') as file:
        json.dump(new_data, file, indent=4)

    print(f"Updated JSON saved to {output_file}")