import tempfile
import gradio as gr
import os
from src.pc.plot_gen.axes_crop import CroppingProcessor

def run_cropping(img_dir, output_dir=None):
    if not img_dir:
        return "No input directory provided from upload step.", gr.update(choices=[]), None

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="crops_")

    try:
        processor = CroppingProcessor()
        processor.create_crops(img_dir, output_dir)
        crops = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])
        return (
            f"Cropping complete.",
            gr.update(choices=crops, value=crops[0] if crops else None),
            output_dir
        )
    except Exception as e:
        return f"Error: {e}", gr.update(choices=[]), None



def show_crop(output_dir, selected_crop):
    if not selected_crop:
        return None
    return os.path.join(output_dir, selected_crop)
