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

def calculate_pixel_positions(df, normalized_columns, height):
    pixel_positions = {}
    for col in normalized_columns:
        col_name = col.strip()
        pixel_positions[col_name] = height - (df[col] * height)
    return pixel_positions

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

def split_json_data(input_json, train_json, valid_json, train_ratio=0.8):
    with open(input_json, 'r') as file:
        data = json.load(file)
    np.random.shuffle(data)
    split_idx = int(len(data) * train_ratio)

    train_data = data[:split_idx]
    valid_data = data[split_idx:]

    with open(train_json, 'w') as train_file:
        json.dump(train_data, train_file, indent=4)

    with open(valid_json, 'w') as valid_file:
        json.dump(valid_data, valid_file, indent=4)

def safe_join(base_dir, file_path):
    if not os.path.isabs(file_path) and not os.path.dirname(file_path):
        return os.path.join(base_dir, file_path)
    return file_path

def round_half_up(x):
    """Round halves up (0.5 -> 1, -0.5 -> 0) for consistent pixel placement."""
    return int(math.floor(float(x) + 0.5))
