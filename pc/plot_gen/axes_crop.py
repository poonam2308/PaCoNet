# axes_crop.py
import xml.etree.ElementTree as ET
import os, io, json
import cairosvg
from PIL import Image, ImageDraw
from typing import Iterable, List
from pc.plot_gen.coordinate_extraction import CoordinateExtraction
from pc.plot_gen.plot_utils import round_half_up


class CroppingProcessor:
    def __init__(self ):
        self.extractor = CoordinateExtraction(normalize_y_to_plot=False)

    def _draw_highlights(self, pil_image, x_coords: Iterable[float], height: int, line_width=0.5, radius=7):
        annotated = pil_image.copy()
        draw = ImageDraw.Draw(annotated)
        for x in x_coords:
            x_int = round_half_up(float(x))
            if x_int < 0 or x_int >= annotated.width:
                continue
            draw.line([(x_int, 0), (x_int, height - 1)], fill=(0, 255, 0), width=line_width)
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
        w = self._to_int_px(root.attrib.get('width'), raster_fallback_w)
        h = self._to_int_px(root.attrib.get('height'), raster_fallback_h)
        return w, h

    def create_crops(self, img_dir: str, output_base_dir: str, eps: float = 0.75):
        os.makedirs(output_base_dir, exist_ok=True)
        files = sorted(os.listdir(img_dir))

        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            fpath = os.path.join(img_dir, fname)
            base, _ = os.path.splitext(fname)

            if ext == ".svg":
                # --- SVG path ---
                svg_path = fpath
                png_bytes = cairosvg.svg2png(url=svg_path)
                img = Image.open(io.BytesIO(png_bytes)).convert("RGB")

                svg_w, svg_h = self._read_svg_size(svg_path, img.width, img.height)
                x_coords = self.extractor.extract_vertical_axes(svg_path)
                x_coords = [x for x in x_coords if 0 <= x < svg_w]

                xs_px: List[int] = sorted(set(round_half_up(x) for x in x_coords))

            elif ext in [".png", ".jpg", ".jpeg"]:
                # --- Raster + metadata JSON ---
                img = Image.open(fpath).convert("RGB")
                json_path = os.path.join(img_dir, f"{base}.json")
                if not os.path.exists(json_path):
                    print(f" No JSON found for {fname}, skipping.")
                    continue

                with open(json_path, "r") as f:
                    data = json.load(f)

                if "vertical_axes" not in data:
                    print(f"JSON for {fname} missing 'vertical_axes', skipping.")
                    continue

                x_coords = data["vertical_axes"]
                xs_px: List[int] = sorted(set(round_half_up(x) for x in x_coords))
                svg_h = img.height

            else:
                continue  # skip unknown file types

            # --- Crop between consecutive axes ---
            for i in range(len(xs_px) - 1):
                l = max(0, xs_px[i])
                r = min(img.width, xs_px[i + 1])
                if r <= l:
                    continue
                crop = img.crop((l, 0, r, img.height))
                crop.save(os.path.join(output_base_dir, f"{base}_crop_{i+1}.png"))
