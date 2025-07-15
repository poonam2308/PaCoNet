import cv2
import numpy as np
import json
import os
from collections import Counter
from PIL import Image
from colorsys import rgb_to_hsv
from sklearn.cluster import DBSCAN

def is_gray_or_white(color):
    r, g, b = color
    if r > 240 and g > 240 and b > 240:
        return True
    if abs(r - g) < 40 and abs(r - b) < 40 and abs(g - b) < 40:
        return True
    return False

def extract_dominant_colors_dbscan(image_path, eps=8, min_samples=120):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
    pixels = img.reshape(-1, 3)
    bg_color = tuple(Counter(map(tuple, pixels)).most_common(1)[0][0])
    pixels = np.array([p for p in pixels if not np.allclose(p, bg_color, atol=10)])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(pixels)
    mask = labels != -1
    filtered_pixels = pixels[mask]
    filtered_labels = labels[mask]
    unique_labels = np.unique(filtered_labels)
    dominant_colors = []
    for label in unique_labels:
        cluster_pixels = filtered_pixels[filtered_labels == label]
        mean_color = np.mean(cluster_pixels, axis=0).astype(int)
        if not is_gray_or_white(mean_color):
            dominant_colors.append(mean_color)

    hsv_colors = []
    for r, g, b in dominant_colors:
        h, s, v = rgb_to_hsv(r / 255, g / 255, b / 255)
        hsv_colors.append((round(h * 360, 2), round(s * 100, 2), round(v * 100, 2)))
    return hsv_colors

def process_image_input(image_input, output_json="dominant_colors.json"):
    """
    Accepts a file path (image) or a directory.
    """
    if os.path.isfile(image_input):
        image_files = [image_input]
    else:
        image_files = [
            os.path.join(image_input, f)
            for f in os.listdir(image_input)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    color_data = {}
    for path in image_files:
        fname = os.path.basename(path)
        colors = extract_dominant_colors_dbscan(path)
        if colors:
            color_data[fname] = colors

    with open(output_json, "w") as f:
        json.dump(color_data, f, indent=4)
    print(f"Saved dominant colors to {output_json}")
    return color_data

