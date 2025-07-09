import gradio as gr
import pandas as pd
import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Ensure reproducibility
random.seed(42)
np.random.seed(42)

sys.path.append(os.path.abspath("."))  # Ensure src/ is importable

# PaCoNet components
from pc.plot_gen.multi_cat import MultiCatPCPGenerator
from pc.plot_gen.single_cat import SingleCatPCPGenerator
from pc.plot_gen.axes_crop import CroppingProcessor
from pc.plot_gen.cat_sep import CategorySeparator
from pc.plot_gen.line_data import LineCoordinateExtractor

# UNet inference components
from pc.models.unet import UNetSD
from pc.data_gen.custom_dataset_unet import CustomTestDatasetSD
from pc.run_epoch_unet import test_unetsd_cluster



# === Initialization ===
multi_gen = MultiCatPCPGenerator(width=600, height=300, show_labels=False)
single_gen = SingleCatPCPGenerator(width=600, height=300, show_labels=False)
cropper = CroppingProcessor()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_model = UNetSD(in_channels=3, out_channels=3).to(device)
unet_model.load_state_dict(torch.load("outputs/chkpt/unet_sd/unet_sd_clusternew_mse_model_epoch50.pth", map_location=device))  # Adjust if needed
unet_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1,1]
])


def clean_output_dir(path):
    if os.path.exists(path):
        for f in os.listdir(path):
            full_path = os.path.join(path, f)
            if os.path.isfile(full_path):
                os.remove(full_path)
            elif os.path.isdir(full_path):
                clean_output_dir(full_path)
                os.rmdir(full_path)
    else:
        os.makedirs(path, exist_ok=True)

# === Step 1: Generate Plot ===
def generate_plot(file, mode):
    clean_output_dir("outputs/plots")
    clean_output_dir("outputs/crops")
    clean_output_dir("outputs/separated")
    clean_output_dir("outputs/denoised")

    df = pd.read_csv(file.name)
    os.makedirs("outputs/plots", exist_ok=True)
    svg_path = "outputs/plots/app_plot.svg"

    if mode == "Multi-category":
        chart, _ = multi_gen.generate_plot(df, filename=svg_path)
    else:
        chart, _, _ = single_gen.generate_plot(df, filename=svg_path)

    return [(svg_path, "Full Plot")], svg_path, mode, file.name, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

# === Step 2: Crop SVGs ===
def crop_plot(svg_path, mode, file_path):
    df = pd.read_csv(file_path)
    cropper.create_crops("outputs/plots", "outputs/crops")

    crops = sorted([
        os.path.join("outputs/crops", f)
        for f in os.listdir("outputs/crops")
        if f.lower().endswith(".png")
    ])

    return [(svg_path, "Full Plot")] + [(p, f"Crop {i+1}") for i, p in enumerate(crops)]

# === Step 3: Separate Crops by Color ===
def separate_categories(file_path):
    extractor = LineCoordinateExtractor(main_dir="outputs/plots")
    extractor.extract_all()
    line_json = extractor.output_file

    sep = CategorySeparator(input_dir="outputs/crops", line_coords_json=line_json)
    sep.separate_by_dbscan(
        output_dir="outputs/separated",
        output_json="outputs/lines.json",
        color_json="outputs/colors.json"
    )

    images = sorted([
        os.path.join("outputs/separated", f)
        for f in os.listdir("outputs/separated")
        if f.endswith(".png")
    ])
    return [(p, f"Category {i+1}") for i, p in enumerate(images)]

# === Step 4: Denoise Categories with UNet ===
def denoise_categories():
    input_dir = "outputs/separated"
    output_dir = "outputs/denoised"
    os.makedirs(output_dir, exist_ok=True)

    dataset = CustomTestDatasetSD(input_dir=input_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    for _ in range(1):
        test_unetsd_cluster(unet_model, loader, device, output_dir)

    images = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".png")
    ])
    return [(p, f"Denoised {i+1}") for i, p in enumerate(images)]

# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("## 📈 PaCoNet — Parallel Coordinates Pipeline + UNet Denoising")

    file_input = gr.File(label="Upload CSV File")
    mode_radio = gr.Radio(["Multi-category", "Single-category"], label="Plot Type", value="Multi-category")
    plot_btn = gr.Button("Generate Plot")

    gallery_plot = gr.Gallery(label="📊 Step 1: Full Plot", columns=3, visible=False)
    crop_btn = gr.Button("Crop This Plot", visible=False)

    gallery_crop = gr.Gallery(label="✂️ Step 2: Cropped Images", columns=3, visible=False)
    sep_btn = gr.Button("Separate Categories", visible=False)

    gallery_sep = gr.Gallery(label="🎨 Step 3: Color Separated", columns=3, visible=False)
    denoise_btn = gr.Button("Denoise Categories", visible=False)

    gallery_denoise = gr.Gallery(label="🧹 Step 4: Denoised Outputs", columns=3, visible=False)

    svg_path_state = gr.State()
    mode_state = gr.State()
    file_path_state = gr.State()

    # Step 1: Generate Plot
    plot_btn.click(
        fn=generate_plot,
        inputs=[file_input, mode_radio],
        outputs=[gallery_plot, svg_path_state, mode_state, file_path_state]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=crop_btn
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=gallery_plot
    )

    # Step 2: Crop Plot
    crop_btn.click(
        fn=crop_plot,
        inputs=[svg_path_state, mode_state, file_path_state],
        outputs=gallery_crop
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=sep_btn
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=gallery_crop
    )

    # Step 3: Separate Categories
    sep_btn.click(
        fn=separate_categories,
        inputs=file_path_state,
        outputs=gallery_sep
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=denoise_btn
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=gallery_sep
    )

    # Step 4: Denoise
    denoise_btn.click(
        fn=denoise_categories,
        outputs=gallery_denoise
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=gallery_denoise
    )


if __name__ == "__main__":
    demo.launch(share=True)
