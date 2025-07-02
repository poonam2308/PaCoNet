import pandas as pd
import numpy as np
import colorsys
import re
import xml.etree.ElementTree as ET
import json


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

def extract_vertical_axes_coords(svg_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    vertical_axes_coords = set()
    transform_pattern = re.compile(r'translate\(([\d\.]+),([\d\.]+)\)')

    for elem in root.iter():
        if elem.tag.endswith('g') and 'role-axis' in elem.attrib.get('class', ''):
            for subelem in elem.iter():
                if subelem.tag.endswith('line') and subelem.attrib.get('y2') and subelem.attrib['y2'] != '0':
                    transform = subelem.attrib.get('transform')
                    if transform:
                        match = transform_pattern.search(transform)
                        if match:
                            x_coord = int(float(match.group(1)))
                            vertical_axes_coords.add(x_coord)

    return sorted(vertical_axes_coords)

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