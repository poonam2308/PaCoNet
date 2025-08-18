# axes_crop.py
import xml.etree.ElementTree as ET
import os
import io
import cairosvg
from PIL import Image, ImageDraw
import math
from typing import Iterable, List

from pc.plot_gen.coordinate_extraction import CoordinateExtraction

def round_half_up(x):
    """Round halves up (0.5 -> 1, -0.5 -> 0) for consistent pixel placement."""
    return int(math.floor(float(x) + 0.5))

class CroppingProcessor:

    def _draw_highlights(self, pil_image, x_coords: Iterable[float], height: int, line_width=0.5, radius=7):
        annotated = pil_image.copy()
        draw = ImageDraw.Draw(annotated)
        for x in x_coords:
            x_int = round_half_up(float(x))
            if x_int < 0 or x_int >= annotated.width:
                continue
            # draw axis guide
            draw.line([(x_int, 0), (x_int, height - 1)], fill=(0, 255, 0), width=line_width)
            # bottom marker
            r = radius
            cx, cy = x_int, height - 1
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 255, 0), width=line_width)
        return annotated

    def _to_int_px(self, v, fallback):
        if v is None:
            return int(fallback)
        s = str(v).strip()
        try:
            return int(float(s[:-2]) if s.endswith("px") else float(s))
        except Exception:
            return int(fallback)

    def _read_svg_size(self, svg_path: str, raster_fallback_w: int, raster_fallback_h: int):
        tree = ET.parse(svg_path)
        root = tree.getroot()
        w = self._to_int_px(root.attrib.get('width'),  raster_fallback_w)
        h = self._to_int_px(root.attrib.get('height'), raster_fallback_h)
        return w, h

    def create_crops(self, img_dir: str, output_base_dir: str, eps: float = 0.75):
        """
        Converts SVG plots to PNGs, saves a highlighted PNG showing vertical axes,
        then crops strictly between consecutive axes and saves the crops.
        """
        os.makedirs(output_base_dir, exist_ok=True)
        svg_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.svg')])

        for svg_name in svg_files:
            svg_path = os.path.join(img_dir, svg_name)
            base, _ = os.path.splitext(svg_name)

            # Render SVG to PNG (bytes) and open with PIL
            png_bytes = cairosvg.svg2png(url=svg_path)
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")

            # SVG's own width/height (fallback to raster size)
            svg_w, svg_h = self._read_svg_size(svg_path, img.width, img.height)

            # 1) Extract vertical axes (deduped)
            extractor = CoordinateExtraction(normalize_y_to_plot=False)
            x_coords = extractor.extract_vertical_axes(svg_path)
            x_coords = [x for x in x_coords if 0 <= x < svg_w]
            # print(x_coords)
            #
            # # 2) Save a debug-highlighted image
            # highlighted = self._draw_highlights(img, x_coords, svg_h, line_width=2, radius=7)
            # highlighted.save(os.path.join(output_base_dir, f"{base}_highlighted.png"))
            #
            # # 3) Crop between consecutive axes, skipping 1px inside each axis
            # pad_in = 0.0
            xs_px: List[int] = [round_half_up(x) for x in x_coords]
            xs_px = sorted(set(xs_px))
            for i in range(len(xs_px) - 1):
                l = max(0, xs_px[i])
                r = min(img.width, xs_px[i + 1])
                if r <= l:
                    continue
                crop = img.crop((l, 0, r, svg_h))
                crop.save(os.path.join(output_base_dir, f"{base}_crop_{i+1}.png"))
