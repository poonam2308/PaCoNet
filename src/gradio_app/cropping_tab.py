import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import gradio as gr
from src.rwd.cropping import crop_images
import json
from .session import SESSION
import shutil


def crop_and_return_images():
    """
    Crop detected regions based on axes, and return gallery items.
    """
    if not all([SESSION["input_path"], SESSION["line_json"], SESSION["cropped_dir"]]):
        return [], [], gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    crop_images(str(SESSION["input_path"]), str(SESSION["line_json"]), str(SESSION["cropped_dir"]))

    image_items, choices = [], []
    for f in sorted(os.listdir(SESSION["cropped_dir"])):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            path = SESSION["cropped_dir"] / f
            image_items.append((str(path), f))
            choices.append(f)

    choices = ["ALL"] + choices
    return (
        image_items,
        gr.update(choices=choices, value=[]),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
    )


def get_base_overlay_with_axes():
    if not SESSION["input_path"] or not SESSION["line_json"]:
        return None

    with open(SESSION["line_json"], "r") as f:
        data = json.load(f)
    if not data:
        return None

    fname = data[0]["image_name"]   # ✅ use consistent filename
    img_path = SESSION["input_path"] / fname
    if not img_path.exists():
        return None

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for entry in data:
        if entry["image_name"] == fname:
            for x in sorted(entry["x_coordinates"]):
                draw.line([(x, 0), (x, img.height)], fill="blue", width=2)
            break

    out_path = Path("outputs/reals/cropped_overlay") / f"{Path(fname).stem}_axes_only.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return str(out_path)



def generate_cropping_overlay(selected_crop_names=None):
    """
    Overlay original image with highlighted or blurred crop regions.
    """
    if not SESSION["input_path"] or not SESSION["line_json"] or not SESSION["cropped_dir"]:
        return None

    with open(SESSION["line_json"], "r") as f:
        data = json.load(f)
    if not data:
        return None

    fname = data[0]["image_name"]   # ✅ use consistent filename
    img_path = SESSION["input_path"] / fname
    if not img_path.exists():
        return None

    base_img = Image.open(img_path).convert("RGB")

    x_coords = []
    for entry in data:
        if entry["image_name"] == fname:
            x_coords = sorted(entry["x_coordinates"])
            break

    crop_files = sorted([f for f in os.listdir(SESSION["cropped_dir"]) if f.endswith(".png")])
    crop_map = {crop_files[i]: (x_coords[i], 0, x_coords[i+1], base_img.height)
                for i in range(min(len(x_coords)-1, len(crop_files)))}

    if "ALL" in selected_crop_names:
        selected_crop_names = list(crop_map.keys())

    output = base_img.copy()
    draw = ImageDraw.Draw(output)
    blur_layer = base_img.filter(ImageFilter.GaussianBlur(radius=5))

    for crop_name, box in crop_map.items():
        if crop_name not in selected_crop_names:
            output.paste(blur_layer.crop(box), box)
        else:
            draw.rectangle(box, outline="red", width=4)

    out_path = Path("outputs/reals/cropped_overlay") / f"{Path(fname).stem}_highlighted_overlay.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(out_path)
    return str(out_path)


def save_selected_images(selected_files):
    if not selected_files:
        return "No files selected."

    if "ALL" in selected_files:
        selected_files = [f for f in os.listdir(SESSION["cropped_dir"]) if
                          f.lower().endswith((".png", ".jpg", ".jpeg"))]

    dest = SESSION["selected_dir"]
    dest.mkdir(parents=True, exist_ok=True)

    for file_name in selected_files:
        src = SESSION["cropped_dir"] / file_name
        shutil.copy(src, dest / file_name)

    return f"Saved {len(selected_files)} images to {dest}"

def select_crop_from_gallery(evt: gr.SelectData):
    """
    Map gallery click (index) to crop filename.
    Returns a single-element list so it syncs with the checkbox group.
    """
    crops = sorted([f for f in os.listdir(SESSION["cropped_dir"]) if f.endswith(".png")])
    if 0 <= evt.index < len(crops):
        return [crops[evt.index]]
    return []