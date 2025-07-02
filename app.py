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

# Initialize processors
multi_gen = MultiCatPCPGenerator(width=600, height=300, show_labels=False)
single_gen = SingleCatPCPGenerator(width=600, height=300, show_labels=False)
cropper = CroppingProcessor()

# Step 1: Generate the plot and save SVG
def generate_plot(file, mode):
    df = pd.read_csv(file.name)
    os.makedirs("outputs/plots", exist_ok=True)
    svg_path = "outputs/plots/app_plot.svg"

    if mode == "Multi-category":
        chart, _ = multi_gen.generate_plot(df, filename=svg_path)
    else:
        chart, _, _ = single_gen.generate_plot(df, filename=svg_path)

    return [(svg_path, "Full Plot")], svg_path, mode, file.name, gr.update(visible=True)

# Step 2: Crop from existing SVG dynamically (no JSON involved)
def crop_plot(svg_path, mode, file_path):
    df = pd.read_csv(file_path)  # Reload to avoid data type issues

    # Use SVG generated earlier — no need to regenerate plot or extract axes separately
    os.makedirs("outputs/crops", exist_ok=True)
    cropper.create_crops("outputs/plots", "outputs/crops")

    # Collect and label crops
    crops = sorted([
        os.path.join("outputs/crops", f)
        for f in os.listdir("outputs/crops")
        if f.endswith(".png")
    ])
    return [(svg_path, "Full Plot")] + [(p, f"Crop {i+1}") for i, p in enumerate(crops)]

# Gradio UI
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

    gallery = gr.Gallery(label="Output: Full Plot + Crops", columns=3)
    svg_path_state = gr.State()
    mode_state = gr.State()
    file_path_state = gr.State()

    # Generate plot and reveal crop button
    plot_btn.click(
        fn=generate_plot,
        inputs=[file_input, mode_radio],
        outputs=[gallery, svg_path_state, mode_state, file_path_state, crop_btn]
    )

    # Crop the current plot using dynamic axes
    crop_btn.click(
        fn=crop_plot,
        inputs=[svg_path_state, mode_state, file_path_state],
        outputs=gallery
    )

if __name__ == "__main__":
    demo.launch()
