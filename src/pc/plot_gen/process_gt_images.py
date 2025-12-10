import os
import re
import json
import shutil

from PIL import Image
import numpy as np
from collections import defaultdict

# ========== PART 1: RENAME IMAGES ==========

def rename_images(src_dir, dst_dir):
    """
    Copy & rename files of the form:
        image_<number>_<cat_name>_crop_<number>.<ext>
    to dst_dir as:
        image_<number>_crop_<number>_<cat_name>.<ext>

    Example:
        image_10_9kixbtF_crop_1.png -> image_10_crop_1_9kixbtF.png
    """
    os.makedirs(dst_dir, exist_ok=True)

    pattern = re.compile(r"^(image_\d+)_(.+)_crop_(\d+)$")

    for fname in os.listdir(src_dir):
        root, ext = os.path.splitext(fname)
        if ext.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        m = pattern.match(root)
        if not m:
            # Doesn't match the pattern; skip
            continue

        img_prefix, cat_name, crop_idx = m.groups()
        new_root = f"{img_prefix}_crop_{crop_idx}_{cat_name}"
        new_fname = new_root + ext

        old_path = os.path.join(src_dir, fname)
        new_path = os.path.join(dst_dir, new_fname)

        if os.path.exists(new_path):
            print(f"[WARN] Target filename already exists, skipping: {new_path}")
            continue

        shutil.copy2(old_path, new_path)


# ========== PART 2: MAKE BLACK/GREY BACKGROUNDS WHITE ==========

def replace_background_with_white(image_path,
                                  dark_brightness_threshold=70,
                                  gray_std_threshold=15):
    """
    Open image, convert dark/grey-ish background pixels to white.

    - dark_brightness_threshold: pixels with average (R,G,B) below this value
      are considered dark.
    - gray_std_threshold: pixels whose R,G,B channels are very similar
      (std dev < gray_std_threshold) are considered "grey-ish".
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Can't open {image_path}: {e}")
        return

    arr = np.array(img, dtype=np.uint8)

    # Compute per-pixel brightness and "colorfulness"
    brightness = arr.mean(axis=2)
    color_std = arr.std(axis=2)

    # Mask for dark-ish gray-ish pixels (background candidates)
    mask = (brightness < dark_brightness_threshold) & (color_std < gray_std_threshold)

    # Set those pixels to white
    arr[mask] = [255, 255, 255]

    img_out = Image.fromarray(arr, mode="RGB")
    img_out.save(image_path)


def whiten_backgrounds_in_dir(images_dir):
    """
    Apply background whitening to all image files in the directory.
    """
    for fname in os.listdir(images_dir):
        root, ext = os.path.splitext(fname)
        if ext.lower() not in [".png", ".jpg", ".jpeg"]:
            continue
        path = os.path.join(images_dir, fname)
        replace_background_with_white(path)


# ========== PART 3: BUILD all_data.json ==========

def build_all_data(json_dir, output_path):
    """
    Read per-image JSONs like your image_1.json and convert them into a
    flat list like your all_data.json:

      [
        { "filename": "image_10_crop_1_9kixbtF.png",
          "lines": [ [x1,y1,x2,y2], ... ] },
        ...
      ]

    For each (crop_k, category_name) pair, we create one entry.
    """
    all_records = []

    for fname in os.listdir(json_dir):
        if not fname.lower().endswith(".json"):
            continue
        if fname == os.path.basename(output_path):
            # Don't re-read an existing all_data.json if present
            continue

        json_path = os.path.join(json_dir, fname)

        with open(json_path, "r") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to parse {json_path}: {e}")
                continue

        # Expect structure like your image_1.json
        base_image_name = data.get("filename")  # e.g. "image_1.png"
        if not base_image_name:
            print(f"[WARN] No 'filename' in {json_path}, skipping.")
            continue

        base_root, _ = os.path.splitext(base_image_name)  # e.g. "image_1"
        lines_by_crop = data.get("lines", {})

        for crop_key, categories in lines_by_crop.items():
            # crop_key like "crop_1"
            try:
                crop_idx = crop_key.split("_")[-1]
            except Exception:
                print(f"[WARN] Unexpected crop key '{crop_key}' in {json_path}, skipping.")
                continue

            if not isinstance(categories, dict):
                print(f"[WARN] 'lines[{crop_key}]' not a dict in {json_path}, skipping.")
                continue

            for cat_name, line_list in categories.items():
                # new filename: image_<n>_crop_<crop_idx>_<cat_name>.png
                new_filename = f"{base_root}_crop_{crop_idx}_{cat_name}.png"

                # line_list is already a list of [x1, y1, x2, y2] (no change)
                record = {
                    "filename": new_filename,
                    "lines": line_list
                }
                all_records.append(record)

    # Save combined data
    # Save combined data (sorted by filename)
    all_records.sort(key=lambda r: r["filename"])

    with open(output_path, "w") as f_out:
        json.dump(all_records, f_out, indent=2, sort_keys=True)

    print(f"Saved {len(all_records)} records into {output_path}")



def group_crops_to_new_json(input_path: str, output_path: str) -> None:
    """
    Reads a JSON file with a list of items like:
      { "filename": "image_10_crop_2_9kixbtF.png", "lines": [...] }
    Groups entries by crop (image_X_crop_Y.png) and writes a new JSON file with:
      { "filename": "image_10_crop_2.png", "lines": [merged lines] }
    """

    # Read input JSON
    with open(input_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    grouped = {}

    for item in items:
        full_name = item.get("filename", "")

        # Split off extension first so we can keep it
        root, ext = os.path.splitext(full_name)
        # remove the random tail, keep "image_X_crop_Y"
        # e.g. "image_10_crop_2_9kixbtF" -> "image_10_crop_2"
        base_root = root.rsplit("_", 1)[0]
        base_name = base_root + ext  # add .png back

        if base_name not in grouped:
            grouped[base_name] = {
                "filename": base_name,
                "lines": []
            }

        grouped[base_name]["lines"].extend(item.get("lines", []))

    # Convert dict -> list
    output_data = list(grouped.values())

    # Write output JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)




def run_rename(image_dir, json_dir, out_image_dir, all_data, all_data_cat , all_data_crop_group):
    """
    image_dir    : directory with original crops (old names)
    json_dir     : directory with per-image JSONs
    out_data     : path to all_data.json to create
    out_image_dir: directory where renamed + whitened images will be saved
    """
    # 1. Copy & rename images into output dir
    print("=== Copying and renaming images to output directory ===")
    # rename_images(image_dir, out_image_dir)

    # 2. Whiten black/grey backgrounds in the output dir
    print("\n=== Whitening backgrounds in output directory ===")
    # whiten_backgrounds_in_dir(out_image_dir)

    # 3. Build all_data.json (filenames match the renamed images)
    print("\n=== Building all_data.json ===")
    # build_all_data(json_dir, all_data)

    print("\nDone.")


# def run_crops(crop_dir, all_data_cat, all_data_crop_group):
#     whiten_backgrounds_in_dir(crop_dir)
#     # 4. this for placing the cat lines with crops.
#     group_crops_to_new_json(all_data_cat, all_data_crop_group)