import json

import gradio as gr
import os
from pathlib import Path
import shutil
import re
import io
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import ImageFilter, ImageDraw, Image, ImageChops

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
def generate_category_overlay(selected_filenames):

    if not selected_filenames or "ALL" in selected_filenames:
        return None

    if not SESSION["input_path"] or not SESSION["line_json"] or not SESSION["cropped_dir"] or not SESSION["separated_dir"]:
        return None

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

    output_images = sorted([p for p in denoised_dir.glob("*.png")])

    SESSION_LOG["steps"].append("Denoising Completed")
    SESSION_LOG["results"]["denoised_files"] = [str(p) for p in output_images]

    gallery_items = [str(p) for p in output_images]
    choice_names = ["ALL"] + [p.name for p in output_images]

    return (
        gallery_items,  # denoise_gallery
        gr.update(visible=True),  # denoise_gallery visible
        f"Denoising complete: {len(output_images)} items.",  # denoise_status
        gr.update(choices=choice_names, visible=True, value=[])  # denoised_selection 👈
    )

def generate_denoised_overlay(selected_filenames):

    if not selected_filenames or "ALL" in selected_filenames:
        return None

    # Pre-reqs
    if not SESSION["input_path"] or not SESSION["line_json"] or not SESSION["cropped_dir"]:
        return None
    denoised_dir = SESSION.get("denoised_dir")
    if not denoised_dir or not denoised_dir.exists():
        return None  # nothing denoised yet

    # --- 1) Group by image_id and keep the first for single-image overlay (like Cropping) ---
    by_img = {}
    for fname in selected_filenames:
        m = re.match(r"(\d+)_crop_(\d+)_cat_(\d+).*\.png", fname)
        if not m:
            continue
        img_id, crop_idx = m.group(1), m.group(2)
        by_img.setdefault(img_id, set()).add(f"{img_id}_crop_{crop_idx}.png")
    if not by_img:
        return None

    image_id, selected_crops = next(iter(by_img.items()))
    orig_fname = f"{image_id}.png"
    img_path = SESSION["input_path"] / orig_fname

    # --- 2) Base blurred image (like cropping overlay) ---
    base_img = Image.open(img_path).convert("RGB")
    blurred_img = base_img.filter(ImageFilter.GaussianBlur(radius=6))
    output = blurred_img.copy()
    draw = ImageDraw.Draw(output)

    # --- 3) Vertical lines / crop boxes for this image ---
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

    # Helper: pick denoised filename that corresponds to a selection name
    # We denoised _wb.png inputs, and the UNet wrote .png outputs with the same base names.
    def resolve_denoised_name(name: str) -> str:
        # Try exact; else strip known suffixes to match denoise output naming
        candidates = [
            name,
            name.replace("_alpha.png", "_wb.png"),
            name.replace("_wb.png", ".png") if name.endswith("_wb.png") else name,
        ]
        for c in candidates:
            p = denoised_dir / c
            if p.exists():
                return c
        # fallback: best-effort (model often preserves stem)
        stem = Path(name).stem
        for p in denoised_dir.glob("*.png"):
            if Path(p).stem == stem or Path(p).stem.startswith(stem):
                return p.name
        return name  # may not exist

    # --- 4) Composite ALL selected denoised categories onto one overlay ---
    for crop_fname, box in crop_map.items():
        if crop_fname not in selected_crops:
            continue

        crop_idx = re.search(r"_crop_(\d+)", crop_fname).group(1)
        # Collect selected categories for this crop (from the user's list)
        matching = [n for n in selected_filenames if n.startswith(f"{image_id}_crop_{crop_idx}_cat_")]
        if not matching:
            continue

        w, h = (box[2] - box[0], box[3] - box[1])
        blended = base_img.crop(box).filter(ImageFilter.GaussianBlur(radius=5)).convert("RGBA")

        for name in matching:
            den_name = resolve_denoised_name(name)
            den_path = denoised_dir / den_name
            if not den_path.exists():
                # Optional: fall back to separated_dir if denoised missing
                sep_path = SESSION["separated_dir"] / name
                if not sep_path.exists():
                    continue
                cat_rgb = Image.open(sep_path).convert("RGB").resize((w, h))
            else:
                cat_rgb = Image.open(den_path).convert("RGB").resize((w, h))

            # Build alpha from “non-white” (display-time only)
            white_bg = Image.new("RGB", (w, h), "white")
            diff = ImageChops.difference(cat_rgb, white_bg).convert("L")
            alpha = diff.point(lambda p: 255 if p > 10 else 0)

            cat_rgba = cat_rgb.copy()
            cat_rgba.putalpha(alpha)
            blended.paste(cat_rgba, (0, 0), mask=alpha)

        output.paste(blended.convert("RGB"), box)

    # Only PNGs
    out_path = Path("outputs/reals/denoised_overlay") / f"{image_id}_overlay.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(out_path)
    return str(out_path)


def trigger_line_prediction(score_threshold):
    denoised_dir = SESSION["denoised_dir"]
    images = [str(denoised_dir / f) for f in os.listdir(denoised_dir) if f.endswith(".png")]
    svg_files, json_files = run_line_prediction_on_images(images, score_threshold=score_threshold)

    # 🔁 NEW: Save directory of JSONs to SESSION
    if json_files:
        json_dir = Path(json_files[0]).parent
        SESSION["json_dir"] = json_dir

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


    SESSION["pred_image_to_json"] = {Path(s).name: j for s, j in zip(sorted_svgs, sorted_jsons)}
    svg_names = [Path(f).name for f in sorted_svgs]

    SESSION_LOG["inputs"]["line_score_threshold"] = score_threshold
    SESSION_LOG["steps"].append("Line Prediction Completed")
    SESSION_LOG["results"]["line_jsons"] = json_files

    return list(sorted_svgs), "\n".join(json_filenames), \
        gr.update(choices=["ALL"] + svg_names, visible=True, value=[]),  \
        gr.update(visible=True)  # show the overlay image area

def generate_predicted_overlay(selected_image_names):
    if not selected_image_names:
        return None

    mapping = SESSION.get("pred_image_to_json", {})
    if not mapping:
        return None

    chosen_imgs = list(mapping.keys()) if "ALL" in selected_image_names \
        else [n for n in selected_image_names if n in mapping]
    if not chosen_imgs:
        return None

    first_json = Path(mapping[chosen_imgs[0]])
    m = re.match(r"(\d+)_crop_", first_json.name)
    if not m:
        return None
    image_id = m.group(1)
    orig_fname = f"{image_id}.png"

    base_img = Image.open(SESSION["input_path"] / orig_fname).convert("RGB")
    blurred = base_img.filter(ImageFilter.GaussianBlur(radius=6))
    output = blurred.copy()  # start blurred everywhere

    # load verticals / axes
    with open(SESSION["line_json"], "r") as f:
        line_data = json.load(f)
    x_coords = []
    for entry in line_data:
        if entry.get("image_name") == orig_fname:
            x_coords = sorted(entry.get("x_coordinates", []))
            break
    if len(x_coords) < 2:
        out_path = Path("outputs/reals/predicted_overlay") / f"{image_id}_pred_overlay.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output.save(out_path)
        return str(out_path)

    H = base_img.height

    # detect crop index base (0 or 1)
    crop_indices_found = []
    for n in os.listdir(SESSION["cropped_dir"]):
        mm = re.search(r"_crop_(\d+)", n)
        if mm: crop_indices_found.append(int(mm.group(1)))
    base0 = 0 if crop_indices_found and min(crop_indices_found) == 0 else 1

    def crop_idx_to_box(crop_idx: int):
        i = crop_idx - base0
        if i < 0 or i >= len(x_coords) - 1:
            return None
        return (x_coords[i], 0, x_coords[i + 1], H)

    # draw axes on top (crisp)
    draw = ImageDraw.Draw(output)
    for x in x_coords:
        draw.line([(x, 0), (x, H)], fill="blue", width=2)

    def svg_to_rgba(svg_path: Path, size_wh):
        try:
            import cairosvg, io
            buf = cairosvg.svg2png(url=str(svg_path), background_color="transparent")
            im = Image.open(io.BytesIO(buf)).convert("RGBA").resize(size_wh, Image.LANCZOS)
        except Exception:
            rgb = Image.open(svg_path).convert("RGB").resize(size_wh, Image.LANCZOS)
            im = rgb.convert("RGBA")
        # ensure non-ink stays transparent (in case background isn’t transparent)
        rgb = im.convert("RGB")
        white = Image.new("RGB", size_wh, "white")
        diff = ImageChops.difference(rgb, white).convert("L")
        alpha = diff.point(lambda p: 255 if p > 10 else 0)
        im.putalpha(alpha)
        return im

    # 🔑 Only sharpen under predicted strokes; everything else stays blurred
    for img_name in chosen_imgs:
        json_path = Path(mapping[img_name])
        mm = re.search(r"_crop_(\d+)", json_path.name)
        if not mm:
            continue
        crop_idx = int(mm.group(1))
        box = crop_idx_to_box(crop_idx)
        if not box:
            continue

        svg_path = json_path.with_suffix(".svg")
        if not svg_path.exists():
            continue

        w, h = box[2] - box[0], box[3] - box[1]
        overlay_rgba = svg_to_rgba(svg_path, (w, h))
        mask = overlay_rgba.split()[-1]  # ink alpha (255 on strokes)

        sharp_crop = base_img.crop(box).convert("RGBA")
        blurred_crop = output.crop(box).convert("RGBA")

        # make region sharp only where mask has strokes
        sharpened_under_lines = Image.composite(sharp_crop, blurred_crop, mask)

        # then paint the colored lines on top
        composited = sharpened_under_lines.copy()
        composited.alpha_composite(overlay_rgba)

        output.paste(composited.convert("RGB"), box)

    out_path = Path("outputs/reals/predicted_overlay") / f"{image_id}_pred_overlay.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(out_path)
    return str(out_path)

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

        stitched_overlay = None
        if svg_file:
            stitched_overlay = generate_stitched_overlay(svg_file, image_id, blur_radius=6, darken_bg=False)

        category_filenames = [[f.stem, ""] for f in stitched_csv_dir.glob("*.csv")]
        SESSION_LOG["inputs"]["stitching_threshold"] = threshold_val
        SESSION_LOG["inputs"]["use_hsv"] = use_hsv_colors
        SESSION_LOG["results"]["final_svg"] = svg_file
        SESSION_LOG["results"]["stitched_csv"] = [str(p) for p in stitched_csv_dir.glob("*.csv")]
        SESSION_LOG["steps"].append("Stitching & Plot Completed")

        return category_filenames, (stitched_overlay or svg_file), "✅ XY Plot generated and saved."

    except Exception as e:
        return [], "", f"❌ Error: {str(e)}"
def generate_stitched_overlay(svg_path: str, image_id: str, blur_radius=6, darken_bg=False):
    """Overlay stitched SVG on blurred input, showing blurred background with sharp lines where SVG has strokes."""
    try:
        import cairosvg
    except Exception:
        cairosvg = None

    orig_fname = f"{image_id}.png"
    base_img = Image.open(SESSION["input_path"] / orig_fname).convert("RGB")
    blurred = base_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    if darken_bg:
        dark_layer = Image.new("RGBA", base_img.size, (0, 0, 0, 80))
        tmp = blurred.convert("RGBA")
        tmp.alpha_composite(dark_layer)
        blurred = tmp.convert("RGB")

    W, H = base_img.size
    if cairosvg:
        png_bytes = cairosvg.svg2png(url=str(svg_path), output_width=W, output_height=H, background_color="transparent")
        overlay = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    else:
        overlay = Image.open(svg_path).convert("RGBA").resize((W, H), Image.LANCZOS)

    # Alpha mask from non-white so background remains transparent
    rgb = overlay.convert("RGB")
    white = Image.new("RGB", (W, H), "white")
    diff = ImageChops.difference(rgb, white).convert("L")
    alpha_mask = diff.point(lambda p: 255 if p > 10 else 0)
    overlay.putalpha(alpha_mask)

    # Make only the SVG stroke areas sharp
    sharp = base_img.convert("RGBA")
    blurred_rgba = blurred.convert("RGBA")
    sharp_under = Image.composite(sharp, blurred_rgba, alpha_mask)

    # Place the colored SVG strokes on top
    composited = sharp_under.copy()
    composited.alpha_composite(overlay)

    # Save final composite over blurred background
    out_path = Path("outputs/reals/stitched_overlay") / f"{image_id}_stitched_overlay.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composited.save(out_path)  # keeps transparency from overlay
    return str(out_path)

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
            category_overlay_img = gr.Image(label="Overlay: Selected Category Highlight", type="filepath", visible=True)

            denoise_selection = gr.CheckboxGroup(label="Select Images for Denoising", choices=["ALL"], visible=False)


        # --- TAB 4: Denoising ---
        with gr.TabItem("4️⃣ Denoising"):
            run_denoise_btn = gr.Button("Run Denoising", visible=False)
            denoise_gallery = gr.Gallery(label="Denoised Images", columns=[4], visible=False)
            denoise_status = gr.Textbox(label="Denoising Status", interactive=False, visible=False)
            denoised_selection = gr.CheckboxGroup(label="Select Denoised Overlays", choices=[], visible=False)
            denoise_overlay_img = gr.Image(
                label="Overlay: Denoised Categories on Original",
                type="filepath",
                visible=True
            )

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

            pred_overlay_selection = gr.CheckboxGroup(
                label="Select Predicted Lines to Overlay",
                choices=[],
                visible=False
            )
            pred_overlay_img = gr.Image(
                label="Overlay: Predicted Lines on Original",
                type="filepath",
                visible=True
            )

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

    denoise_selection.change(
        fn=generate_category_overlay,
        inputs=[denoise_selection],
        outputs=[category_overlay_img]
    )

    run_denoise_btn.click(
        fn=run_denoising,
        inputs=[denoise_selection],  # existing category selection
        outputs=[denoise_gallery, denoise_gallery, denoise_status, denoised_selection]
    )

    denoised_selection.change(
        fn=generate_denoised_overlay,
        inputs=[denoised_selection],
        outputs=[denoise_overlay_img]
    )

    predict_btn.click(
        fn=trigger_line_prediction,
        inputs=[line_threshold_slider],
        outputs=[svg_gallery, json_output, pred_overlay_selection, pred_overlay_img]
    )

    pred_overlay_selection.change(
        fn=generate_predicted_overlay,
        inputs=[pred_overlay_selection],
        outputs=[pred_overlay_img]
    )

    run_stitch_btn.click(
        fn=run_stitching_from_prediction,
        inputs=[threshold_input, use_hsv_checkbox],
        outputs=[csv_list_output, svg_viewer, csv_status]
    )

    download_log_btn.click(fn=save_session_log, outputs=[log_file_output])

demo.launch()

