from PIL import ImageFilter, Image, ImageChops
import io
from pathlib import Path
from rwd.st import convert_json_to_csv, stitch_xy_coordinates
from rwd.pt import process_stitched_xy_csvs
from .session import SESSION, SESSION_LOG

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
        overlay = overlay.resize((W, H), Image.LANCZOS)

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
