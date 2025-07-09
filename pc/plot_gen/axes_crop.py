import xml.etree.ElementTree as ET
import os
import torch
import io
import cairosvg
from PIL import Image
from pc.plot_gen.plot_utils import extract_number, extract_vertical_axes_coords


class CroppingProcessor:
    def create_crops(self, img_dir, output_base_dir):
        """
        Converts SVG plots to PNGs and crops them between vertical axes.
        """
        os.makedirs(output_base_dir, exist_ok=True)

        image_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith('.svg')],
            key=extract_number
        )

        for idx, img_file in enumerate(image_files):
            img_path = os.path.join(img_dir, img_file)
            image_name, _ = os.path.splitext(img_file)

            # Convert SVG to PNG image
            png_image_data = cairosvg.svg2png(url=img_path)
            image = Image.open(io.BytesIO(png_image_data)).convert("RGB")

            # Extract dimensions from SVG
            tree = ET.parse(img_path)
            root = tree.getroot()
            original_width = int(root.attrib['width'])
            original_height = int(root.attrib['height'])

            # Dynamically extract vertical axis positions
            x_coords = extract_vertical_axes_coords(img_path)
            x_coords = torch.tensor(x_coords, dtype=torch.float)

            for crop_idx in range(len(x_coords) - 1):
                left_x = int(x_coords[crop_idx])
                right_x = int(x_coords[crop_idx + 1])

                # Crop and resize
                cropped_image = image.crop((left_x + 6, 0, right_x + 4, original_height))
                crop_width = right_x - left_x
                cropped_image = cropped_image.resize((crop_width, original_height))

                # Save crop
                save_path = os.path.join(output_base_dir, f"{image_name}_crop_{crop_idx + 1}.png")
                cropped_image.save(save_path)
