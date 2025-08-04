import json

import gradio as gr
import os
from pathlib import Path
import shutil
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import ImageFilter, ImageDraw, Image

from rwd.color_extraction import process_image_input
from rwd.axes_detection import detect_vertical_axes
from rwd.cropping import crop_images
from rwd.category_separation import process_images_separation

# UNet inference components
from pc.models.unet import UNetSD
from pc.data_gen.custom_dataset_unet import CustomTestDatasetSD
from pc.run_epoch_unet import test_unetsd_cluster
import torch
from dhlp.line_prediction import run_line_prediction_on_images
from rwd.stitching_interface import run_stitching_from_json_dir
from rwd.st import convert_json_to_csv, stitch_xy_coordinates
from rwd.pt import process_stitched_xy_csvs


from rwd.plot_redesigner import generate_allcat_hsvplots_for_directory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_model = UNetSD(in_channels=3, out_channels=3).to(device)
unet_model.load_state_dict(torch.load("outputs/chkpt/unet_sd/unet_sd_clusternew_mse_model_epoch50.pth", map_location=device))  # Adjust if needed
unet_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1,1]
])

SESSION_LOG = {
    "inputs": {},
    "steps": [],
    "results": {}
}

# Store session-specific paths
SESSION = {
    "input_path": None,
    "line_output_dir": None,
    "line_json": None,
    "cropped_dir": None,
    "selected_dir": None,
    "separated_dir": None,
    "denoised_dir": None,
}


def save_uploaded_file(file_obj, tmp_dir):
    ext = Path(file_obj).suffix.lower()
    new_name = Path(file_obj).stem + ".png" if ext != ".png" else Path(file_obj).name
    dest = tmp_dir / new_name

    img = Image.open(file_obj).convert("RGB")
    img.save(dest)
    return str(dest)
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
    tmp_dir = output_root

    # Clear previously created folders
    folders_to_clear = ["input", "lined_images", "cropped", "selected",
                        "separated", "denoised","redesigned"]
    for folder in folders_to_clear:
        full_path = tmp_dir / folder
        if full_path.exists():
            shutil.rmtree(full_path)


    input_path = tmp_dir / "input"
    os.makedirs(input_path, exist_ok=True)

    if isinstance(file_or_folder, list):
        for path in file_or_folder:
            save_uploaded_file(path, input_path)
    else:
        save_uploaded_file(file_or_folder, input_path)

    SESSION["input_path"] = input_path
    SESSION["line_output_dir"] = tmp_dir / "lined_images"
    SESSION["line_json"] = tmp_dir / "verticals.json"
    SESSION["cropped_dir"] = tmp_dir / "cropped"
    SESSION["selected_dir"] = tmp_dir / "selected"

    process_image_input(str(input_path), output_json=str(tmp_dir / "colors.json"))
    detect_vertical_axes(
        str(input_path),
        str(SESSION["line_output_dir"]),
        str(SESSION["line_json"]),
        apertureSize=aperture_size,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
        min_spacing = min_spacing,
        left_edge_thresh = left_edge_thresh,
        right_edge_thresh= right_edge_thresh,
        method ="combined"
    )
    SESSION["all_detected_coords"] = extract_coords_per_image(SESSION["line_json"])

    output_images = []
    for img_file in os.listdir(SESSION["line_output_dir"]):
        img_path = SESSION["line_output_dir"] / img_file
        img = Image.open(img_path).convert("RGB")
        output_images.append(img)

    SESSION_LOG["inputs"]["uploaded_files"] = [str(p) for p in file_or_folder] if isinstance(file_or_folder,
                                                                                             list) else [
        str(file_or_folder)]
    SESSION_LOG["inputs"]["line_detection_params"] = {
        "aperture_size": aperture_size,
        "min_line_length": min_line_length,
        "max_line_gap": max_line_gap
    }
    SESSION_LOG["steps"].append("Axes Detection Completed")

    return output_images

def overlay_vertical_lines_on_image(image_path, json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    fname = Path(image_path).name
    for entry in data:
        if entry["image_name"] == fname:
            for x in entry["x_coordinates"]:
                draw.line([(x, 0), (x, img.height)], fill="red", width=2)
            break

    out_path = Path("outputs/reals/lined_overlay") / fname
    os.makedirs(out_path.parent, exist_ok=True)
    img.save(out_path)
    return str(out_path)


def get_original_input_image():
    if not SESSION["input_path"] or not SESSION["line_json"]:
        return None

    files = sorted(os.listdir(SESSION["input_path"]))
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = SESSION["input_path"] / f
            return overlay_vertical_lines_on_image(str(img_path), str(SESSION["line_json"]))
    return None



def crop_and_return_images():
    if not all([SESSION["input_path"], SESSION["line_json"], SESSION["cropped_dir"]]):
        return [], [], gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    crop_images(str(SESSION["input_path"]), str(SESSION["line_json"]), str(SESSION["cropped_dir"]))

    image_items = []
    choices = []
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
    """Draw only vertical lines (no crop highlights yet)."""
    if not SESSION["input_path"] or not SESSION["line_json"]:
        return None

    fname = sorted(os.listdir(SESSION["input_path"]))[0]
    img_path = SESSION["input_path"] / fname
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    with open(SESSION["line_json"], "r") as f:
        data = json.load(f)

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
    if not SESSION["input_path"] or not SESSION["line_json"] or not SESSION["cropped_dir"]:
        return None

    if selected_crop_names is None or "ALL" in selected_crop_names:
        selected_crop_names = []  # Treat as no specific selection (just show outlines)

    # Get the original image path
    fname = sorted(os.listdir(SESSION["input_path"]))[0]
    img_path = SESSION["input_path"] / fname
    base_img = Image.open(img_path).convert("RGB")

    with open(SESSION["line_json"], "r") as f:
        data = json.load(f)

    x_coords = []
    for entry in data:
        if entry["image_name"] == fname:
            x_coords = sorted(entry["x_coordinates"])
            break

    # Map crop filenames to box coords
    crop_files = sorted([f for f in os.listdir(SESSION["cropped_dir"]) if f.lower().endswith(".png")])
    crop_map = {}  # crop_name → (box coords)
    for i in range(min(len(x_coords)-1, len(crop_files))):
        box = (x_coords[i], 0, x_coords[i+1], base_img.height)
        crop_map[crop_files[i]] = box

    output = base_img.copy()
    draw = ImageDraw.Draw(output)
    blur_layer = base_img.filter(ImageFilter.GaussianBlur(radius=5))

    for crop_name, box in crop_map.items():
        if selected_crop_names and crop_name not in selected_crop_names:
            # Blur unselected crops
            region = blur_layer.crop(box)
            output.paste(region, box)
        elif crop_name in selected_crop_names:
            draw.rectangle(box, outline="red", width=4)
        elif not selected_crop_names:
            draw.rectangle(box, outline="green", width=2)

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


def run_category_separation(method, top_k):
    input_dir = SESSION["selected_dir"]
    output_dir = Path("outputs/reals/separated")
    output_json = Path("outputs/reals/separated/separation_data.json")
    SESSION["dominant_colors_json"] = output_json

    if not input_dir.exists() or not any(input_dir.iterdir()):
        return [], "No selected crops found."

    # Run the separation
    process_images_separation(
        input_dir=input_dir,
        output_dir=output_dir,
        output_json=output_json,
        method='topk',
        top_k=3  # ← This is the value that controls how many top hues to extract
    )

    image_items = []
    denoise_choices = []

    for f in sorted(os.listdir(output_dir)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            full_path = str(output_dir / f)
            image_items.append((full_path, f))
            denoise_choices.append(f)

    choices = ["ALL"] + denoise_choices

    SESSION["separated_dir"] = output_dir
    SESSION["denoised_dir"] = Path("outputs/reals/denoised")

    status = f"Category separation complete: {len(image_items)} items."

    SESSION_LOG["inputs"]["category_method"] = method
    SESSION_LOG["inputs"]["top_k"] = top_k
    SESSION_LOG["steps"].append("Category Separation Completed")

    return (
        image_items,
        gr.update(visible=True),
        status,
        gr.update(choices=choices, visible=True, value=[]),
        gr.update(visible=True),
        gr.update(visible=True),
    )

def run_denoising(selected_files):
    separated_dir = SESSION["separated_dir"]
    denoised_dir = Path("outputs/reals/denoised")
    denoised_dir.mkdir(parents=True, exist_ok=True)

    # If "ALL" selected, copy everything from separated_dir
    input_dir = Path("outputs/reals/temp_unet_input")
    shutil.rmtree(input_dir, ignore_errors=True)
    input_dir.mkdir(parents=True)

    all_files = {f.name: f for f in separated_dir.glob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]}

    files_to_copy = (
        all_files.values() if "ALL" in selected_files
        else [all_files[f] for f in selected_files if f in all_files]
    )

    for src in files_to_copy:
        shutil.copy(src, input_dir / src.name)

    dataset = CustomTestDatasetSD(input_dir=input_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    for _ in range(1):
        test_unetsd_cluster(unet_model, loader, device, denoised_dir)

    images = sorted([
        os.path.join(denoised_dir, f)
        for f in os.listdir(denoised_dir)
        if f.endswith(".png")
    ])
    # Return paths for gallery
    output_images = sorted(denoised_dir.glob("*"))
    SESSION_LOG["steps"].append("Denoising Completed")
    SESSION_LOG["results"]["denoised_files"] = [str(p) for p in output_images]

    return (
        [str(p) for p in output_images],
        gr.update(visible=True),
        f"Denoising complete: {len(output_images)} items."
    )

def trigger_line_prediction(score_threshold):
    denoised_dir = SESSION["denoised_dir"]
    images = [str(denoised_dir / f) for f in os.listdir(denoised_dir) if f.endswith(".png")]
    svg_files, json_files = run_line_prediction_on_images(images, score_threshold=score_threshold)

    # 🔁 NEW: Save directory of JSONs to SESSION
    if json_files:
        json_dir = Path(json_files[0]).parent
        SESSION["json_dir"] = json_dir

    # Optional: clean sort
    import re
    def extract_cat_crop_key(filename):
        match = re.search(r'_crop_(\d+)_cat_(\d+)', filename)
        if match:
            crop_num = int(match.group(1))
            cat_num = int(match.group(2))
            return (cat_num, crop_num)
        return (float('inf'), float('inf'))

    sorted_pairs = sorted(zip(svg_files, json_files), key=lambda pair: extract_cat_crop_key(Path(pair[1]).name))
    sorted_svgs, sorted_jsons = zip(*sorted_pairs)
    json_filenames = [Path(f).name for f in sorted_jsons]
    SESSION_LOG["inputs"]["line_score_threshold"] = score_threshold
    SESSION_LOG["steps"].append("Line Prediction Completed")
    SESSION_LOG["results"]["line_jsons"] = json_files

    return list(sorted_svgs), "\n".join(json_filenames)

from rwd.plot_redesigner import generate_allcat_hsvplots_for_directory

def run_stitching_from_prediction(threshold_val, use_hsv_colors):
    json_dir = SESSION.get("json_dir")
    if not json_dir:
        raise ValueError("JSON directory not set. Run line prediction first.")

    try:
        # dfs_by_cat, (combined_filename, combined_df) = run_stitching_from_json_dir(json_dir, threshold_val)

        # Step 1: Convert JSONs to CSV
        convert_json_to_csv(json_dir)

        # Step 2: Extract image ID from filenames (assumes consistent naming)
        sample_json = next(json_dir.glob("*.json"), None)
        if not sample_json:
            return [], "", "❌ No JSON files found."

        import re
        match = re.match(r"(\d+)_crop_", sample_json.name)
        if not match:
            return [], "", "❌ Could not extract image ID from filename."
        image_id = match.group(1)

        # Step 3: Stitch coordinates
        combined_df = stitch_xy_coordinates(json_dir, image_id, crop_width=224.0, threshold_distance=threshold_val)
        combined_filename = f"{image_id}_stitched_xy"

        # Save stitched CSV
        stitched_csv_dir = json_dir / "stitched_csv"
        stitched_csv_dir.mkdir(exist_ok=True)
        stitched_csv_path = stitched_csv_dir / f"{combined_filename}.csv"
        combined_df.to_csv(stitched_csv_path, index=False)
        # hsv_json_path = SESSION["separated_dir"] / "hue_summary.json" if use_hsv_colors else None
        hsv_json_path = SESSION["dominant_colors_json"] if use_hsv_colors else None

        process_stitched_xy_csvs(
            input_dir=stitched_csv_dir,
            output_dir=stitched_csv_dir,
            crop_width_val=224.0,
            hsv_json_path=str(hsv_json_path) if use_hsv_colors and hsv_json_path else None
        )

        output_svg_path = list(stitched_csv_dir.glob("*.svg"))
        svg_file = str(output_svg_path[0]) if output_svg_path else None

        category_filenames = [[f.stem, ""] for f in stitched_csv_dir.glob("*.csv")]
        SESSION_LOG["inputs"]["stitching_threshold"] = threshold_val
        SESSION_LOG["inputs"]["use_hsv"] = use_hsv_colors
        SESSION_LOG["results"]["final_svg"] = svg_file
        SESSION_LOG["results"]["stitched_csv"] = [str(p) for p in stitched_csv_dir.glob("*.csv")]
        SESSION_LOG["steps"].append("Stitching & Plot Completed")

        return category_filenames, svg_file, "✅ XY Plot generated and saved."

    except Exception as e:
        return [], "", f"❌ Error: {str(e)}"

def save_session_log():
    log_path = Path("outputs/reals/session_log.json")
    with open(log_path, "w") as f:
        json.dump(SESSION_LOG, f, indent=2)
    return str(log_path)


# --- Gradio Interface ---
with gr.Blocks(title="PaCoNet - Data Extraction and Plot Redesign") as demo:
    gr.Markdown("## 📊 PaCoNet - Data Extraction and Plot Redesign")



    with gr.Tabs():
        # --- TAB 1: Line Detection ---
        with gr.TabItem("1️⃣ Axes Detection"):
            with gr.Row():
                file_input = gr.File(file_types=[".png", ".jpg", ".jpeg"], file_count="multiple", label="Upload Image(s)")
            with gr.Row():
                aperture = gr.Slider(3, 7, step=2, value=5, label="Canny Aperture Size")
                min_len = gr.Slider(10, 150, step=2, value=20, label="Min Line Length")
                max_gap = gr.Slider(1, 50, step=1, value=1, label="Max Line Gap")
                min_spacing_slider = gr.Slider(5, 100, step=1, value=20, label="Minimum Spacing Between Axes (px)")
                left_thresh_slider = gr.Slider(0.0, 0.2, step=0.01, value=0.03,
                                               label="Left Edge Ignore Threshold (fraction)")
                right_thresh_slider = gr.Slider(0.8, 1.0, step=0.01, value=0.95,
                                                label="Right Edge Ignore Threshold (fraction)")

            run_btn = gr.Button("Run Detection")
            img_output = gr.Gallery(label="Detected Images with Lines", columns=[3], height=400)
            with gr.Row():
                image_selector = gr.Dropdown(choices=[], label="Select Image for Coordinate Filtering")
                coord_selector = gr.CheckboxGroup(choices=[], label="Select X-Coordinates to Keep")
                save_coords_btn = gr.Button("💾 Save Selected Coordinates")
                save_coords_status = gr.Textbox(label="Save Status", interactive=False)

        # --- TAB 2: Cropping ---
        with gr.TabItem("2️⃣ Cropping") as cropping_tab:
            # original_input_img = gr.Image(label="Original Image", type="filepath")
            crop_btn = gr.Button("Crop Images After Detection")
            cropped_output = gr.Gallery(label="Cropped Images", columns=[4], height=300)
            cropped_selection = gr.CheckboxGroup(label="Select Cropped Files to Save", choices=[], visible=False)
            save_selected_btn = gr.Button("Save Selected Crops", visible=False)
            save_result_text = gr.Textbox(label="Save Result", interactive=False, visible=False)

            # 🔍 New overlay visualization in cropping tab
            crop_overlay_img = gr.Image(label="Overlay with Crop Highlight", type="filepath")

        # --- TAB 3: Category Separation ---
        with gr.TabItem("3️⃣ Category Separation") as sep_tab:
            method_dropdown = gr.Dropdown(
                choices=["peaks", "topk"],
                value="topk",
                label="Hue Separation Method"
            )
            topk_slider = gr.Slider(
                minimum=1,
                maximum=10,
                step=1,
                value=3,
                label="Top K Dominant Hues (Only used if method='topk')"
            )

            sep_btn = gr.Button("Run Category Separation", visible=False)
            sep_result_text = gr.Textbox(label="Separation Status", interactive=False, visible=False)
            sep_gallery = gr.Gallery(label="Separated Categories", columns=[4], visible=False)
            denoise_selection = gr.CheckboxGroup(label="Select Images for Denoising", choices=["ALL"], visible=False)

        # --- TAB 4: Denoising ---
        with gr.TabItem("4️⃣ Denoising"):
            run_denoise_btn = gr.Button("Run Denoising", visible=False)
            denoise_gallery = gr.Gallery(label="Denoised Images", columns=[4], visible=False)
            denoise_status = gr.Textbox(label="Denoising Status", interactive=False, visible=False)

        with gr.TabItem("5️⃣ Line Prediction"):
            # yaml_input = gr.File(label="YAML Config", file_types=[".yaml"])
            line_threshold_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.5,
                label="Line Confidence Threshold"
            )

            predict_btn = gr.Button("Run Line Prediction")
            svg_gallery = gr.Gallery(label="Line SVGs", columns=3)
            json_output = gr.Textbox(label="JSON Outputs", lines=10)

        with gr.TabItem("6️⃣ Stitch & View CSVs"):
            threshold_input = gr.Slider(minimum=1.0, maximum=50.0, step=1.0, value=10.0, label="Matching Threshold")
            use_hsv_checkbox = gr.Checkbox(label="Use Custom HSV Colors", value=True)
            run_stitch_btn = gr.Button("Run Stitching from Predictions")
            csv_list_output = gr.Dataframe(
                headers=["Stitched CSV Filename", " "],
                datatype=["str", "str"],
                interactive=False,
                label="Stitched CSV Files (per Category)"
            )
            svg_viewer = gr.Image(label="Redesigned Plot", type="filepath")
            csv_status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        download_log_btn = gr.Button("📥 Download Session Log")
        log_file_output = gr.File(label="Downloadable Session Summary")



    # --- Hook up logic ---
    run_btn.click(fn=process_input,
                  inputs=[file_input, aperture, min_len, max_gap, min_spacing_slider, left_thresh_slider,
                          right_thresh_slider],
                  outputs=[img_output]
                  ).then(
        fn=update_coordinate_selector,
        inputs=[],
        outputs=[image_selector, coord_selector]
    )

    image_selector.change(fn=update_coords_for_image,
                          inputs=[image_selector],
                          outputs=[coord_selector])

    save_coords_btn.click(fn=save_selected_axes,
                          inputs=[image_selector, coord_selector],
                          outputs=[save_coords_status])

    crop_btn.click(
        fn=crop_and_return_images,
        outputs=[
            cropped_output, cropped_selection,
            cropped_selection, save_selected_btn, save_result_text, sep_btn
        ]
    ).then(
        fn=get_base_overlay_with_axes,
        outputs=[crop_overlay_img]
    )

    cropped_selection.change(
        fn=generate_cropping_overlay,
        inputs=[cropped_selection],
        outputs=[crop_overlay_img]
    )

    save_selected_btn.click(fn=save_selected_images,
                            inputs=[cropped_selection],
                            outputs=[save_result_text])

    sep_btn.click(
        fn=run_category_separation,
        inputs=[method_dropdown, topk_slider],
        outputs=[
            sep_gallery,
            sep_gallery,
            sep_result_text,
            denoise_selection,
            run_denoise_btn,
            denoise_status
        ]
    )

    run_denoise_btn.click(
        fn=run_denoising,
        inputs=[denoise_selection],
        outputs=[denoise_gallery, denoise_gallery, denoise_status]
    )

    predict_btn.click(
        fn=trigger_line_prediction,
        inputs=[line_threshold_slider],
        outputs=[svg_gallery, json_output]
    )
    run_stitch_btn.click(
        fn=run_stitching_from_prediction,
        inputs=[threshold_input, use_hsv_checkbox],
        outputs=[csv_list_output, svg_viewer, csv_status]
    )

    download_log_btn.click(fn=save_session_log, outputs=[log_file_output])

demo.launch()

