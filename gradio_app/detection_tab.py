import os, json, shutil
from pathlib import Path
from PIL import Image, ImageDraw
from rwd.color_extraction import process_image_input
from rwd.axes_detection import detect_vertical_axes
from .session import SESSION, SESSION_LOG
from .io_utils import save_uploaded_file
import gradio as gr


def extract_coords_per_image(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return {item["image_name"]: item["x_coordinates"] for item in data}

def save_selected_axes(image_name, selected_coords):
    if not selected_coords:
        return "❌ No coordinates selected."

    selected_coords = [int(x) for x in selected_coords]
    original_data = extract_coords_per_image(SESSION["line_json"])
    filtered_data = []

    for name, coords in original_data.items():
        if name == image_name:
            filtered_data.append({
                "image_name": name,
                "x_coordinates": sorted(selected_coords)
            })
        else:
            filtered_data.append({
                "image_name": name,
                "x_coordinates": coords
            })

    filtered_path = SESSION["line_json"].parent / "filtered_verticals.json"
    with open(filtered_path, "w") as f:
        json.dump(filtered_data, f, indent=4)

    SESSION["line_json"] = filtered_path
    return f"✅ Saved filtered coordinates for {image_name}."

# def get_coordinate_selector_data():
#     coord_map = SESSION.get("all_detected_coords", {})
#     if not coord_map:
#         return [], []
#
#     first_img = list(coord_map.keys())[0]
#     x_coords = coord_map[first_img]
#     return list(coord_map.keys()), x_coords
#
#
# def get_coords_for_image(image_name):
#     coord_map = SESSION.get("all_detected_coords", {})
#     return coord_map.get(image_name, [])


def update_coordinate_selector():
    coord_map = SESSION.get("all_detected_coords", {})
    if not coord_map:
        return gr.update(choices=[]), gr.update(choices=[], value=[])

    first_img = list(coord_map.keys())[0]
    x_coords = coord_map[first_img]
    return gr.update(choices=list(coord_map.keys()), value=first_img), gr.update(choices=x_coords, value=x_coords)

def update_coords_for_image(image_name):
    coord_map = SESSION.get("all_detected_coords", {})
    coords = coord_map.get(image_name, [])
    return gr.update(choices=coords, value=coords)

def process_input(file_or_folder, aperture_size, min_line_length, max_line_gap,
                  min_spacing, left_edge_thresh, right_edge_thresh):

    output_root = Path("outputs/reals")
    output_root.mkdir(exist_ok=True, parents=True)

    # Clear old outputs
    for folder in ["input", "lined_images", "cropped", "selected", "separated", "denoised", "redesigned"]:
        full_path = output_root / folder
        if full_path.exists():
            shutil.rmtree(full_path)

    input_path = output_root / "input"
    input_path.mkdir(exist_ok=True)

    if isinstance(file_or_folder, list):
        for path in file_or_folder:
            save_uploaded_file(path, input_path)
    else:
        save_uploaded_file(file_or_folder, input_path)

    SESSION.update({
        "input_path": input_path,
        "line_output_dir": output_root / "lined_images",
        "line_json": output_root / "verticals.json",
        "cropped_dir": output_root / "cropped",
        "selected_dir": output_root / "selected"
    })

    process_image_input(str(input_path), output_json=str(output_root / "colors.json"))
    detect_vertical_axes(
        str(input_path),
        str(SESSION["line_output_dir"]),
        str(SESSION["line_json"]),
        apertureSize=aperture_size,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
        min_spacing=min_spacing,
        left_edge_thresh=left_edge_thresh,
        right_edge_thresh=right_edge_thresh,
        method="combined"
    )

    SESSION["all_detected_coords"] = extract_coords_per_image(SESSION["line_json"])

    output_images = []
    for img_file in os.listdir(SESSION["line_output_dir"]):
        img = Image.open(SESSION["line_output_dir"] / img_file).convert("RGB")
        output_images.append(img)

    SESSION_LOG["inputs"]["uploaded_files"] = [str(file_or_folder)] if not isinstance(file_or_folder, list) else [str(p) for p in file_or_folder]
    SESSION_LOG["inputs"]["line_detection_params"] = {
        "aperture_size": aperture_size,
        "min_line_length": min_line_length,
        "max_line_gap": max_line_gap
    }
    SESSION_LOG["steps"].append("Axes Detection Completed")

    return output_images
