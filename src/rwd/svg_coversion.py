import os
import cairosvg

def convert_svg_to_png(image_folder, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for file_name in os.listdir(image_folder):
        if file_name.lower().endswith(".svg"):
            svg_path = os.path.join(image_folder, file_name)
            png_path = os.path.join(save_dir, os.path.splitext(file_name)[0] + ".png")
            cairosvg.svg2png(url=svg_path, write_to=png_path, background_color='white')
            print(f"Converted: {file_name} -> {png_path}")
    print("SVG to PNG conversion complete.")
