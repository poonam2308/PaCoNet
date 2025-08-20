import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import gradio as gr

from pc.plot_gen.axes_crop import CroppingProcessor
from .session import SESSION
import shutil
from gradio.events import SelectData



def crop_and_return_images():
    """
    Run CroppingProcessor on the active image and return gallery items.
    """
    active = SESSION.get("active_image")
    if not active:
        return [], [], gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    state = SESSION["images"][active]
    img_dir = str(state["input_path"].parent)     # directory with the image/json/svg
    cropped_dir = state["cropped_dir"]
    cropped_dir.mkdir(parents=True, exist_ok=True)

    # Run processor
    processor = CroppingProcessor()
    processor.create_crops(img_dir, str(cropped_dir))

    # Build gallery
    image_items, choices = [], []
    for f in sorted(os.listdir(cropped_dir)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            path = cropped_dir / f
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
    """
    Return the original image with vertical axes drawn (from metadata_json).
    """
    active = SESSION.get("active_image")
    if not active:
        return None
    state = SESSION["images"][active]
    img_path = state["input_path"]
    json_path = state["metadata_json"]

    if not img_path.exists() or not json_path.exists():
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    # vertical_axes may be inside "vertical_axes" or detection-style [{"x_coordinates":...}]
    if isinstance(data, dict) and "vertical_axes" in data:
        x_coords = data["vertical_axes"]
    elif isinstance(data, list) and "x_coordinates" in data[0]:
        x_coords = data[0]["x_coordinates"]
    else:
        return None

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for x in sorted(x_coords):
        draw.line([(x, 0), (x, img.height)], fill="blue", width=2)

    out_path = Path("outputs/reals/cropped_overlay") / f"{Path(active).stem}_axes_only.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return str(out_path)


def generate_cropping_overlay(selected_crop_names=None):
    """
    Overlay image with blurred background + red rectangle on selected crops.
    """
    active = SESSION.get("active_image")
    if not active:
        return None
    state = SESSION["images"][active]
    img_path = state["input_path"]
    json_path = state["metadata_json"]
    cropped_dir = state["cropped_dir"]

    if not img_path.exists() or not json_path.exists() or not cropped_dir.exists():
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "vertical_axes" in data:
        x_coords = sorted(data["vertical_axes"])
    elif isinstance(data, list) and "x_coordinates" in data[0]:
        x_coords = sorted(data[0]["x_coordinates"])
    else:
        return None

    base_img = Image.open(img_path).convert("RGB")
    crop_files = sorted([f for f in os.listdir(cropped_dir) if f.endswith(".png")])
    crop_map = {crop_files[i]: (x_coords[i], 0, x_coords[i + 1], base_img.height)
                for i in range(min(len(x_coords) - 1, len(crop_files)))}

    if selected_crop_names and "ALL" in selected_crop_names:
        selected_crop_names = list(crop_map.keys())

    output = base_img.copy()
    draw = ImageDraw.Draw(output)
    blur_layer = base_img.filter(ImageFilter.GaussianBlur(radius=5))

    for crop_name, box in crop_map.items():
        if not selected_crop_names or crop_name not in selected_crop_names:
            output.paste(blur_layer.crop(box), box)
        else:
            draw.rectangle(box, outline="red", width=4)

    out_path = Path("outputs/reals/cropped_overlay") / f"{Path(active).stem}_highlighted_overlay.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(out_path)
    return str(out_path)


def save_selected_images(selected_files):
    active = SESSION.get("active_image")
    if not active:
        return "No active image selected."
    state = SESSION["images"][active]
    cropped_dir = state["cropped_dir"]
    dest = state["selected_dir"]
    dest.mkdir(parents=True, exist_ok=True)

    if not selected_files:
        return "No files selected."

    if "ALL" in selected_files:
        selected_files = [f for f in os.listdir(cropped_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for file_name in selected_files:
        src = cropped_dir / file_name
        shutil.copy(src, dest / file_name)

    return f"Saved {len(selected_files)} images to {dest}"


def select_crop_from_gallery(evt: SelectData):
    active = SESSION.get("active_image")
    if not active:
        return []
    state = SESSION["images"][active]
    cropped_dir = state["cropped_dir"]
    crops = sorted([f for f in os.listdir(cropped_dir) if f.endswith(".png")])
    if 0 <= evt.index < len(crops):
        return [crops[evt.index]]
    return []
