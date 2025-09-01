import json, shutil
from pathlib import Path
from PIL import Image, ImageDraw
from src.rwd.color_extraction import process_image_input
from src.rwd.axes_detection import detect_vertical_axes
from src.rwd.cropping import crop_images
from .session import SESSION, SESSION_LOG, init_image_session
from .io_utils import save_uploaded_file
import gradio as gr



def update_coords_for_image(image_name):
    coord_map = SESSION.get("all_detected_coords", {})
    coords = coord_map.get(image_name, [])
    return gr.update(choices=coords, value=coords)

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
    return f"Saved filtered coordinates for {image_name}."

def toggle_ui(use_json_choice):
    if use_json_choice == "Yes":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    else:
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)



def extract_coords_from_metadata(json_file):
    """
    Reads a metadata JSON file and extracts vertical axes.
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    filename = data.get("filename")
    coords = data.get("vertical_axes", [])
    return filename, coords


def process_input(files, json_files, aperture_size, min_line_length, max_line_gap,
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

    # Save uploaded images
    image_map = {}
    if isinstance(files, list):
        for path in files:
            saved = save_uploaded_file(path, input_path)
            image_map[Path(saved).name] = Path(saved)
    else:
        saved = save_uploaded_file(files, input_path)
        image_map[Path(saved).name] = Path(saved)

    # Map JSON files to image filenames
    json_map = {}
    if json_files:
        if not isinstance(json_files, list):
            json_files = [json_files]
        for jf in json_files:
            with open(jf, "r") as f:
                data = json.load(f)
            fname = data.get("filename")
            if fname:
                json_map[fname] = Path(jf)

    detected_images = []

    # Process each uploaded image
    for img_name, img_path in image_map.items():
        json_path = json_map.get(img_name)

        # Init session entry
        init_image_session(img_name, img_path, json_path, output_root)

        if json_path:
            # Use vertical_axes from JSON
            with open(json_path, "r") as f:
                data = json.load(f)
            coords = data.get("vertical_axes", [])

            # Draw preview with vertical lines
            im = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(im)
            for x in coords:
                draw.line([(x, 0), (x, im.height)], fill="red", width=2)

            lined_dir = output_root / "lined_images"
            lined_dir.mkdir(parents=True, exist_ok=True)
            out_path = lined_dir / img_name
            im.save(out_path)

            detected_images.append(im)

            # Crop immediately using provided axes
            cropped_dir = SESSION["images"][img_name]["cropped_dir"]
            cropped_dir.mkdir(parents=True, exist_ok=True)

            # Convert to detection-style JSON format
            detection_style_json = [{
                "image_name": img_name,
                "x_coordinates": coords
            }]
            with open(json_path, "w") as f:
                json.dump(detection_style_json, f, indent=4)

            crop_images(str(img_path), str(json_path), str(cropped_dir))

        else:
            # Run detection if no metadata JSON
            line_output_dir = output_root / "lined_images"
            line_output_dir.mkdir(parents=True, exist_ok=True)

            generated_json = output_root / f"{Path(img_name).stem}_verticals.json"

            process_image_input(str(img_path), output_json=str(output_root / "colors.json"))
            detect_vertical_axes(
                str(img_path),
                str(line_output_dir),
                str(generated_json),
                apertureSize=aperture_size,
                minLineLength=min_line_length,
                maxLineGap=max_line_gap,
                min_spacing=min_spacing,
                left_edge_thresh=left_edge_thresh,
                right_edge_thresh=right_edge_thresh,
                method="combined"
            )

            # Update session with generated JSON
            SESSION["images"][img_name]["metadata_json"] = generated_json

            im = Image.open(line_output_dir / img_name).convert("RGB")
            detected_images.append(im)

    # Set the first image as active
    if image_map:
        first_img = list(image_map.keys())[0]
        SESSION["active_image"] = first_img

    SESSION_LOG["inputs"]["uploaded_files"] = [str(p) for p in image_map.values()]
    if json_files:
        SESSION_LOG["inputs"]["metadata_jsons"] = [str(j) for j in json_files]
    else:
        SESSION_LOG["inputs"]["line_detection_params"] = {
            "aperture_size": aperture_size,
            "min_line_length": min_line_length,
            "max_line_gap": max_line_gap
        }

    SESSION_LOG["steps"].append("Detection+Cropping Completed")

    return detected_images
