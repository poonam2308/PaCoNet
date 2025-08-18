import os
from pathlib import Path
import re
from PIL import Image, ImageDraw, ImageFilter, ImageChops
import gradio as gr
from rwd.category_separation import process_images_separation
import json
from .session import SESSION, SESSION_LOG


def run_category_separation(method, top_k):
    """
    Run hue-based category separation.
    """
    input_dir = SESSION["selected_dir"]
    output_dir = Path("outputs/reals/separated")
    output_json = output_dir / "separation_data.json"
    SESSION["dominant_colors_json"] = output_json

    if not input_dir.exists() or not any(input_dir.iterdir()):
        return [], "No selected crops found."

    process_images_separation(
        input_dir=input_dir,
        output_dir=output_dir,
        output_json=output_json,
        method=method,
        top_k=top_k
    )

    image_items, choices = [], []
    for f in sorted(os.listdir(output_dir)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            full_path = str(output_dir / f)
            image_items.append((full_path, f))
            choices.append(f)

    SESSION["separated_dir"] = output_dir
    SESSION["denoised_dir"] = Path("outputs/reals/denoised")

    SESSION_LOG["inputs"]["category_method"] = method
    SESSION_LOG["inputs"]["top_k"] = top_k
    SESSION_LOG["steps"].append("Category Separation Completed")

    return (
        image_items,
        gr.update(visible=True),
        f"Category separation complete: {len(image_items)} items.",
        gr.update(choices=["ALL"] + choices, visible=True, value=[]),
        gr.update(visible=True),
        gr.update(visible=True),
    )


def generate_category_overlay(selected_filenames):
    if not SESSION["input_path"] or not SESSION["line_json"] or not SESSION["cropped_dir"] or not SESSION[
        "separated_dir"]:
        return None

        #  if ALL: select everything
        # --- CASE 1: No selection → return only blurred image with axes ---
    if not selected_filenames:
        # get first image
        fname = sorted(os.listdir(SESSION["input_path"]))[0]
        img_path = SESSION["input_path"] / fname
        base_img = Image.open(img_path).convert("RGB")
        blurred_img = base_img.filter(ImageFilter.GaussianBlur(radius=6))
        draw = ImageDraw.Draw(blurred_img)

        # draw axes
        with open(SESSION["line_json"], "r") as f:
            line_data = json.load(f)
        for entry in line_data:
            if entry["image_name"] == fname:
                for x in sorted(entry["x_coordinates"]):
                    draw.line([(x, 0), (x, base_img.height)], fill="blue", width=2)
                break

        out_path = Path("outputs/reals/category_overlay") / f"{Path(fname).stem}_overlay_blurred.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        blurred_img.save(out_path)
        return str(out_path)

        # --- CASE 2: ALL selected → expand to all separated files ---
    if "ALL" in selected_filenames:
        selected_filenames = [f for f in os.listdir(SESSION["separated_dir"]) if f.endswith(".png")]


    # --- 1) Group selection by image_id and keep the first image_id (single overlay like Cropping) ---
    by_img = {}
    for fname in selected_filenames:
        m = re.match(r"(\d+)_crop_(\d+)_cat_(\d+)\.png", fname)
        if not m:
            continue
        img_id, crop_idx = m.group(1), m.group(2)
        by_img.setdefault(img_id, set()).add(f"{img_id}_crop_{crop_idx}.png")

    if not by_img:
        return None

    image_id, selected_crops = next(iter(by_img.items()))  # show one base image overlay
    orig_fname = f"{image_id}.png"
    img_path = SESSION["input_path"] / orig_fname

    # --- 2) Base image + blur (like cropping overlay) ---
    base_img = Image.open(img_path).convert("RGB")
    blurred_img = base_img.filter(ImageFilter.GaussianBlur(radius=6))
    output = blurred_img.copy()
    draw = ImageDraw.Draw(output)

    # --- 3) Get vertical lines / crop boxes for this image ---
    with open(SESSION["line_json"], "r") as f:
        line_data = json.load(f)

    x_coords = []
    for entry in line_data:
        if entry["image_name"] == orig_fname:
            x_coords = sorted(entry["x_coordinates"])
            break

    for x in x_coords:
        draw.line([(x, 0), (x, base_img.height)], fill="blue", width=2)

    crop_files = sorted([f for f in os.listdir(SESSION["cropped_dir"]) if f.endswith(".png")])
    crop_map = {}
    for i, crop_fname in enumerate(crop_files):
        if i >= len(x_coords) - 1:
            continue
        box = (x_coords[i], 0, x_coords[i + 1], base_img.height)
        crop_map[crop_fname] = box

    # --- 4) Paste ALL selected categories for this image onto one overlay ---
    for crop_fname, box in crop_map.items():
        if crop_fname not in selected_crops:
            # keep blurred background for non-selected crops (like Cropping tab)
            continue

        # find all separated category files that match this crop index
        crop_idx = re.search(r"_crop_(\d+)", crop_fname).group(1)
        matching = [name for name in selected_filenames if name.startswith(f"{image_id}_crop_{crop_idx}_cat_")]
        if not matching:
            continue

        # base crop area (blur underneath)
        w, h = (box[2] - box[0], box[3] - box[1])
        blurred_crop = base_img.crop(box).filter(ImageFilter.GaussianBlur(radius=5))

        # stack all picked categories for this crop onto the blurred crop
        blended = blurred_crop.convert("RGBA")
        for sep_name in matching:
            sep_path = SESSION["separated_dir"] / sep_name
            if not sep_path.exists():
                continue

            cat_rgb = Image.open(sep_path).convert("RGB").resize((w, h))

            # derive transparency from non-white (display-time only)
            white_bg = Image.new("RGB", (w, h), "white")
            diff = ImageChops.difference(cat_rgb, white_bg).convert("L")
            alpha = diff.point(lambda p: 255 if p > 10 else 0)

            cat_rgba = cat_rgb.copy()
            cat_rgba.putalpha(alpha)
            blended.paste(cat_rgba, (0, 0), mask=alpha)

        # paste the combined crop back into the full image
        output.paste(blended.convert("RGB"), box)

    out_path = Path("outputs/reals/category_overlay") / f"{image_id}_overlay.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(out_path)
    return str(out_path)

