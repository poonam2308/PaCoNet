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

def extract_dominant_colors_dbscan_old(image_path, eps=8, min_samples=120):
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

    # Convert to normalized HSV and label as cat1/cat2/...
    # cat_colors = {}
    # for i, (r, g, b) in enumerate(dominant_colors, start=1):
    #     h, s, v = rgb_to_hsv(r / 255, g / 255, b / 255)  # h,s,v already 0..1
    #     cat_colors[f"cat{i}"] = {
    #         "h": round(h, 2),
    #         "s": round(s, 2),
    #         "v": round(v, 2),
    #     }
    #
    # return cat_colors

    # --- only keep distinct H (normalized 0..1), force s=v=1 ---
    hues = []
    for r, g, b in dominant_colors:
        h, s, v = rgb_to_hsv(r / 255, g / 255, b / 255)  # h is 0..1
        hues.append(h)

    # Sort for stable cat1/cat2/... order
    hues.sort()

    # Deduplicate hues: treat hues within this tolerance as "same"
    H_TOL = 0.03  # ~11 degrees; adjust if you want stricter/looser
    unique_hues = []
    for h in hues:
        if not unique_hues or min(abs(h - uh) for uh in unique_hues) > H_TOL:
            unique_hues.append(h)

    cat_colors = {}
    for i, h in enumerate(unique_hues, start=1):
        cat_colors[f"cat{i}"] = {"h": round(h, 2), "s": 1, "v": 1}

    return cat_colors



def extract_prominent_hues_auto(
    image_path: str,
    resize_factor: int = 4,
    bins: int = 72,                 # 5 degrees per bin
    white_v_thresh: float = 0.95,
    white_s_thresh: float = 0.12,
    s_min: float = 0.22,
    v_min: float = 0.20,
    merge_tol: float = 0.06,        # ~22 degrees in 0..1
    peak_rel_min: float = 0.18,     # keep peaks >= 18% of the strongest
    max_cats: int = 8               # safety cap
):
    img = cv2.imread(image_path)
    if img is None:
        return {}

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize_factor and resize_factor > 1:
        img = cv2.resize(img, (img.shape[1] // resize_factor, img.shape[0] // resize_factor))

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    H = hsv[..., 0] / 179.0
    S = hsv[..., 1] / 255.0
    V = hsv[..., 2] / 255.0

    # mask out background-ish and non-color pixels
    # not_white = ~((V >= white_v_thresh) & (S <= white_s_thresh))
    # colorful = (S >= s_min) & (V >= v_min)
    # mask = not_white & colorful
    # if not np.any(mask):
    #     return {}
    #
    # h = H[mask]
    # s = S[mask]
    # v = V[mask]
    #
    # # weight vivid pixels higher (helps thin but saturated lines still count)
    # weights = (s * (0.5 + 0.5 * v))

    not_white = ~((V >= white_v_thresh) & (S <= white_s_thresh))

    # Keep pastel colors but reject gray-ish stuff
    chroma = S * V
    mask = not_white & (V >= 0.25) & (chroma >= 0.06) & (S >= 0.06)

    if not np.any(mask):
        return {}

    h = H[mask]
    s = S[mask]
    v = V[mask]

    # Gentler weighting so pastels still compete
    weights = (np.sqrt(s) * v)

    hist, _ = np.histogram(h, bins=bins, range=(0.0, 1.0), weights=weights)

    # find local maxima peaks
    peaks = []
    for i in range(bins):
        left = hist[(i - 1) % bins]
        mid  = hist[i]
        right= hist[(i + 1) % bins]
        if mid > left and mid >= right and mid > 0:
            center = (i + 0.5) / bins
            peaks.append((mid, center))

    if not peaks:
        return {}

    # sort by peak strength
    peaks.sort(key=lambda x: x[0], reverse=True)
    strongest = peaks[0][0]

    def circ_dist(a, b):
        d = abs(a - b)
        return min(d, 1.0 - d)

    # merge nearby peaks (weighted)
    merged = []  # [weight, hue]
    for w, hue in peaks:
        placed = False
        for j in range(len(merged)):
            w2, h2 = merged[j]
            if circ_dist(hue, h2) <= merge_tol:
                merged[j][1] = (h2 * w2 + hue * w) / (w2 + w)
                merged[j][0] = w2 + w
                placed = True
                break
        if not placed:
            merged.append([w, float(hue)])

    # re-sort merged by prominence
    merged.sort(key=lambda x: x[0], reverse=True)

    # auto-stop rule: only keep peaks that are large enough vs strongest
    kept = []
    for w, hue in merged:
        if len(kept) >= max_cats:
            break
        if w < strongest * peak_rel_min:
            continue
        kept.append((w, hue))

    # fallback: if auto rule was too strict, keep at least 1
    if not kept:
        kept = [tuple(merged[0])]

    out = {}
    for i, (_, hue) in enumerate(kept, start=1):
        out[f"cat{i}"] = {"h": round(hue, 2), "s": 1, "v": 1}
    return out

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
        # colors = extract_dominant_colors_dbscan(path)
        colors = extract_prominent_hues_auto(path)
        if colors:
            color_data[fname] = colors

    with open(output_json, "w") as f:
        json.dump(color_data, f, indent=4)
    print(f"Saved dominant colors to {output_json}")
    return color_data

