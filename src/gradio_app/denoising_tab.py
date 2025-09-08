from pathlib import Path
import os, shutil
from torch.utils.data import DataLoader
from torchvision import transforms
import gradio as gr
from src.pc.models.unet import UNetSD
from src.pc.data_gen.custom_dataset_unet import CustomTestDatasetSD
from src.pc.run_epoch_unet import test_unetsd_cluster
import torch
import re
import json
from PIL import Image, ImageDraw, ImageChops, ImageFilter
from .session import SESSION, SESSION_LOG

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_model = UNetSD(in_channels=3, out_channels=3).to(device)
unet_model.load_state_dict(torch.load(
    "outputs/chkpt/unet_sd/unet_sd_clusternew_mse_model_epoch50.pth",
    map_location=device
))
unet_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


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
    # Pre-reqs
    if not SESSION["input_path"] or not SESSION["line_json"] or not SESSION["cropped_dir"]:
        return None
    denoised_dir = SESSION.get("denoised_dir")
    if not denoised_dir or not denoised_dir.exists():
        return None  # nothing denoised yet
    # --- CASE 1: No selection → just blurred background + axes ---
    if not selected_filenames:
        with open(SESSION["line_json"], "r") as f:
            line_data = json.load(f)
        if not line_data:
            return None

        fname = line_data[0]["image_name"]  # ✅ consistent filename

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

        out_path = Path("outputs/reals/denoise_overlay") / f"{Path(fname).stem}_overlay_blurred.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        blurred_img.save(out_path)
        return str(out_path)
        # --- CASE 2: ALL selected → expand to all denoised files ---
    if "ALL" in selected_filenames:
        selected_filenames = [f for f in os.listdir(SESSION["denoised_dir"]) if f.endswith(".png")]
    # --- 1) Group by image_id and keep the first for single-image overlay (like Cropping) ---
    by_img = {}
    for fname in selected_filenames:
        stem = Path(fname).stem
        parts = stem.rsplit("_crop_", 1)
        if len(parts) != 2:
            continue
        img_id, rest = parts
        crop_idx = rest.split("_")[0]
        by_img.setdefault(img_id, set()).add(f"{img_id}_crop_{crop_idx}.png")

    if not by_img:
        return None

    image_id, selected_crops = next(iter(by_img.items()))
    with open(SESSION["line_json"], "r") as f:
        line_data = json.load(f)
    orig_entry = next((e for e in line_data if e["image_name"].startswith(str(image_id))), None)
    if not orig_entry:
        return None
    orig_fname = orig_entry["image_name"]  # ✅ exact original name

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
        matching = [n for n in selected_filenames if n.startswith(f"{image_id}_crop_{crop_idx}_")]
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

def select_denoised_from_gallery(evt: gr.SelectData):
    """
    Map gallery click (index) to denoised filename.
    Single-selection: returns a one-element list.
    """
    denoised_dir = SESSION.get("denoised_dir")
    if not denoised_dir or not denoised_dir.exists():
        return []
    denoised_files = sorted([f.name for f in denoised_dir.glob("*.png")])
    if 0 <= evt.index < len(denoised_files):
        return [denoised_files[evt.index]]
    return []