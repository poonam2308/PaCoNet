import gradio as gr
import pandas as pd
import os
import sys
import random
import numpy as np

random.seed(42)
np.random.seed(42)

sys.path.append(os.path.abspath("."))  # Ensure src/ is importable

from src.plot_gen.multi_cat import MultiCatPCPGenerator
from src.plot_gen.single_cat import SingleCatPCPGenerator
from src.plot_gen.axes_crop import CroppingProcessor
from src.plot_gen.cat_sep import CategorySeparator
from src.plot_gen.line_data import LineCoordinateExtractor

# Initialize processors
multi_gen = MultiCatPCPGenerator(width=600, height=300, show_labels=False)
single_gen = SingleCatPCPGenerator(width=600, height=300, show_labels=False)
cropper = CroppingProcessor()

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


# Step 1: Generate the plot and save SVG
def generate_plot(file, mode):
    # Clean previous outputs
    clean_output_dir("outputs/plots")
    clean_output_dir("outputs/crops")
    clean_output_dir("outputs/separated")


    df = pd.read_csv(file.name)
    os.makedirs("outputs/plots", exist_ok=True)
    svg_path = "outputs/plots/app_plot.svg"

    if mode == "Multi-category":
        chart, _ = multi_gen.generate_plot(df, filename=svg_path)
    else:
        chart, _, _ = single_gen.generate_plot(df, filename=svg_path)

    return [(svg_path, "Full Plot")], svg_path, mode, file.name, gr.update(visible=True), gr.update(visible=False)

# Step 2: Crop the plot into vertical slices
def crop_plot(svg_path, mode, file_path):
    df = pd.read_csv(file_path)
    os.makedirs("outputs/crops", exist_ok=True)
    cropper.create_crops("outputs/plots", "outputs/crops")

    crops = sorted([
        os.path.join("outputs/crops", f)
        for f in os.listdir("outputs/crops")
        if f.lower().endswith(".png") and os.path.isfile(os.path.join("outputs/crops", f))
    ])

    return [(svg_path, "Full Plot")] + [(p, f"Crop {i+1}") for i, p in enumerate(crops)]

# Step 3: Separate cropped images by color categories
def separate_categories(file_path):
    extractor = LineCoordinateExtractor(main_dir="outputs/plots")
    extractor.extract_all()  # Saves to outputs/plots/alldata.json by default
    line_json = extractor.output_file

    os.makedirs("outputs/separated", exist_ok=True)
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

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## 📈 PaCoNet — Parallel Coordinates Plot")

    file_input = gr.File(label="Upload CSV File")
    mode_radio = gr.Radio(
        choices=["Multi-category", "Single-category"],
        label="Plot Type",
        value="Multi-category"
    )

    plot_btn = gr.Button("Generate Plot")
    crop_btn = gr.Button("Crop This Plot", visible=False)
    sep_btn = gr.Button("Separate Categories", visible=False)

    gallery = gr.Gallery(label="Output: Full Plot + Crops", columns=3)
    svg_path_state = gr.State()
    mode_state = gr.State()
    file_path_state = gr.State()

    # Step 1: Generate Plot
    plot_btn.click(
        fn=generate_plot,
        inputs=[file_input, mode_radio],
        outputs=[gallery, svg_path_state, mode_state, file_path_state, crop_btn, sep_btn]
    )

    # Step 2: Crop the plot
    crop_btn.click(
        fn=crop_plot,
        inputs=[svg_path_state, mode_state, file_path_state],
        outputs=gallery
    ).then(
        fn=lambda mode: gr.update(visible=(mode == "Multi-category")),
        inputs=mode_state,
        outputs=sep_btn
    )

    # Step 3: Separate Categories (only for Multi-category)
    sep_btn.click(
        fn=separate_categories,
        inputs=file_path_state,
        outputs=gallery
    )

if __name__ == "__main__":
    demo.launch()
