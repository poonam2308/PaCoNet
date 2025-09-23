import os, re, json

import numpy as np
from PIL import ImageFilter, ImageDraw, Image, ImageChops
from pathlib import Path
from src.dhlp.line_prediction import run_line_prediction_on_images_all, run_line_prediction_on_images_mask
import cairosvg, io
import gradio as gr
from .session import SESSION, SESSION_LOG


def _sort_pairs(svg_files, json_files):
    def extract_cat_crop_key(filename):
        m = re.match(r'.+_crop_(\d+)_([^_]+)_(pre|mask|post|mask_post)\.json$', filename)
        if m:
            crop_idx = int(m.group(1))
            suffix = m.group(2)
            return (crop_idx, suffix)
        return (float('inf'), '')
    pairs = sorted(zip(svg_files, json_files),
                   key=lambda p: extract_cat_crop_key(Path(p[1]).name))
    svgs, jsons = zip(*pairs) if pairs else ([], [])
    return list(svgs), list(jsons)


def _prep_gallery_payload(svg_files, json_files, target_mapping_key):
    if not svg_files or not json_files:
        return [], "", gr.update(choices=[], visible=False), gr.update(visible=True)
    svg_names = [Path(s).name for s in svg_files]
    json_names = [Path(j).name for j in json_files]

    # Remember directory of JSONs (for stitching later)
    json_dir = Path(json_files[0]).parent
    SESSION["json_dir"] = json_dir  # ok to overwrite; both pre/post live together

    # Map SVG filename -> JSON path for overlay
    SESSION[target_mapping_key] = {Path(s).name: j for s, j in zip(svg_files, json_files)}

    return (
        svg_files,
        "\n".join(json_names),
        gr.update(choices=["ALL"] + svg_names, visible=True, value=[]),
        gr.update(visible=True)
    )

def trigger_line_prediction_both(score_threshold):
    result = run_line_prediction_on_images_mask(
        [str(p) for p in (SESSION.get("denoised_dir") or Path("outputs/reals/denoised")).glob("*.png")],
        score_threshold=score_threshold
    )

    # New API (dict with pre & post)
    if isinstance(result, dict) and "pre" in result and "post" in result:
        pre_svgs, pre_jsons = _sort_pairs(result["pre"]["svgs"], result["pre"]["jsons"])
        post_svgs, post_jsons = _sort_pairs(result["post"]["svgs"], result["post"]["jsons"])

        SESSION_LOG["inputs"]["line_score_threshold"] = score_threshold
        SESSION_LOG["steps"].append("Line Prediction Completed (pre+post)")
        SESSION_LOG["results"]["line_jsons_pre"] = pre_jsons
        SESSION_LOG["results"]["line_jsons_post"] = post_jsons

        # Return 8 outputs: pre(4) + post(4)
        return (
            *_prep_gallery_payload(pre_svgs, pre_jsons, "pred_image_to_json_pre"),
            *_prep_gallery_payload(post_svgs, post_jsons, "pred_image_to_json_post"),
        )

    # Back-compat: old API (post only)
    svg_files, json_files = result
    post_svgs, post_jsons = _sort_pairs(svg_files, json_files)
    SESSION_LOG["inputs"]["line_score_threshold"] = score_threshold
    SESSION_LOG["steps"].append("Line Prediction Completed (post only)")
    SESSION_LOG["results"]["line_jsons_post"] = post_jsons

    # Pre section is empty; post section filled
    return (
        # pre
        [], "", gr.update(choices=[], visible=False), gr.update(visible=True),
        # post
        *_prep_gallery_payload(post_svgs, post_jsons, "pred_image_to_json_post"),
    )

def trigger_line_prediction_selected(score_threshold, kinds):
    # Normalize selection; fall back to all 4 if nothing was ticked
    kinds = set(kinds or ["pre", "mask", "post", "mask_post"])

    denoised_dir_path = SESSION.get("denoised_dir") or Path("outputs/reals/denoised")
    if not denoised_dir_path.exists() or not any(denoised_dir_path.glob("*.png")):
        # return 4 empty sections in the same order: pre, mask, post, mask_post
        empty = ([], "", gr.update(choices=[], visible=False), gr.update(visible=True))
        return (*empty, *empty, *empty, *empty)

    # 👉 pass `kinds` down so we only compute what was selected
    _ = run_line_prediction_on_images_all(
        [str(p) for p in denoised_dir_path.glob("*.png")],
        score_threshold=score_threshold,
        kinds=list(kinds),                   # NEW
    )

    # Collect whatever was generated
    pred_dir = Path("outputs/reals/redesigned")
    if not pred_dir.exists():
        empty = ([], "", gr.update(choices=[], visible=False), gr.update(visible=True))
        return (*empty, *empty, *empty, *empty)

    jsons_by_kind = {
        "pre": list(pred_dir.glob("*_pre.json")),
        "mask": list(pred_dir.glob("*_mask.json")),
        "post": list(pred_dir.glob("*_post.json")),
        "mask_post": list(pred_dir.glob("*_mask_post.json")),
    }

    # Save to SESSION_LOG + NPZs (only for the ones present)
    for kind, json_files in jsons_by_kind.items():
        if json_files:
            SESSION_LOG["results"][f"line_jsons_{kind}"] = [str(j) for j in json_files]
            npz_files = save_predictions_as_npz(json_files, kind)
            SESSION_LOG["results"][f"pred_npz_files_{kind}"] = npz_files

    def prep_display(kind):
        jsons = SESSION_LOG["results"].get(f"line_jsons_{kind}", [])
        svgs = [j.replace(".json", ".svg") for j in jsons]
        return _prep_gallery_payload(svgs, jsons, f"pred_image_to_json_{kind}")

    SESSION_LOG["inputs"]["line_score_threshold"] = score_threshold
    SESSION_LOG["steps"].append(f"Line Prediction Completed ({', '.join(sorted(kinds))})")

    # Always return 4 blocks in fixed order; unselected ones will be empty
    return (
        *prep_display("pre"),
        *prep_display("mask"),
        *prep_display("post"),
        *prep_display("mask_post"),
    )


def trigger_line_prediction_all(score_threshold):
    denoised_dir_path = SESSION.get("denoised_dir") or Path("outputs/reals/denoised")

    if not denoised_dir_path.exists() or not any(denoised_dir_path.glob("*.png")):
        return [], "", gr.update(choices=[], visible=False), gr.update(visible=True), \
            [], "", gr.update(choices=[], visible=False), gr.update(visible=True), \
            [], "", gr.update(choices=[], visible=False), gr.update(visible=True), \
            [], "", gr.update(choices=[], visible=False), gr.update(visible=True)

    # Call the line prediction function
    # NOTE: This function likely saves files to disk but returns nothing or an empty result
    result = run_line_prediction_on_images_all(
        [str(p) for p in denoised_dir_path.glob("*.png")],
        score_threshold=score_threshold
    )

    # After the prediction runs, manually find the generated JSON files
    # This is the crucial fix to get the file paths
    pred_dir = Path("outputs/reals/redesigned")
    if not pred_dir.exists():
        print("Prediction directory not found.")
        return [], "", gr.update(choices=[], visible=False), gr.update(visible=True), \
            [], "", gr.update(choices=[], visible=False), gr.update(visible=True), \
            [], "", gr.update(choices=[], visible=False), gr.update(visible=True), \
            [], "", gr.update(choices=[], visible=False), gr.update(visible=True)

    # Find the JSON files for each 'kind' of prediction
    jsons_by_kind = {
        "pre": list(pred_dir.glob("*_pre.json")),
        "mask": list(pred_dir.glob("*_mask.json")),
        "post": list(pred_dir.glob("*_post.json")),
        "mask_post": list(pred_dir.glob("*_mask_post.json")),
    }

    # Populate the SESSION_LOG and save the NPZ files
    for kind, json_files in jsons_by_kind.items():
        if json_files:
            SESSION_LOG["results"][f"line_jsons_{kind}"] = [str(j) for j in json_files]
            npz_files = save_predictions_as_npz(json_files, kind)
            SESSION_LOG["results"][f"pred_npz_files_{kind}"] = npz_files
        else:
            print(f"No JSONs generated for '{kind}'. Skipping NPZ saving.")

    # Now, process the results for display in the gallery
    def prep_display(kind):
        jsons = SESSION_LOG["results"].get(f"line_jsons_{kind}", [])
        svgs = [j.replace(".json", ".svg") for j in jsons]  # Assumes SVGs are also generated

        # NOTE: You'll need to manually sort them if necessary, or ensure `line_prediction` does
        return _prep_gallery_payload(svgs, jsons, f"pred_image_to_json_{kind}")

    SESSION_LOG["inputs"]["line_score_threshold"] = score_threshold
    SESSION_LOG["steps"].append("Line Prediction Completed (4 variants)")

    return (
        *prep_display("pre"),
        *prep_display("mask"),
        *prep_display("post"),
        *prep_display("mask_post"),
    )
def trigger_line_prediction_all_old(score_threshold):
    result = run_line_prediction_on_images_all(
        [str(p) for p in (SESSION.get("denoised_dir") or Path("outputs/reals/denoised")).glob("*.png")],
        score_threshold=score_threshold
    )

    if not isinstance(result, dict):
        return [], "", gr.update(choices=[], visible=False), gr.update(visible=True), \
               [], "", gr.update(choices=[], visible=False), gr.update(visible=True), \
               [], "", gr.update(choices=[], visible=False), gr.update(visible=True), \
               [], "", gr.update(choices=[], visible=False), gr.update(visible=True)

    # Sort and unpack each of the 4 types
    def prep(kind, target_mapping_key):
        svgs, jsons = _sort_pairs(result[kind]["svgs"], result[kind]["jsons"])
        SESSION_LOG["results"][f"line_jsons_{kind}"] = jsons
        return _prep_gallery_payload(svgs, jsons, target_mapping_key)

    SESSION_LOG["inputs"]["line_score_threshold"] = score_threshold
    SESSION_LOG["steps"].append("Line Prediction Completed (4 variants)")

    for kind in ["pre", "mask", "post", "mask_post"]:
        jsons = SESSION_LOG["results"].get(f"line_jsons_{kind}", [])
        if jsons:
            npz_files = save_predictions_as_npz(jsons, kind)
            SESSION_LOG["results"][f"pred_npz_files_{kind}"] = npz_files
        else:
            # Provide feedback if a particular prediction kind failed to generate JSONs
            print(f"No JSONs generated for '{kind}'. Skipping NPZ saving.")

    return (
        *prep("pre", "pred_image_to_json_pre"),
        *prep("mask", "pred_image_to_json_mask"),
        *prep("post", "pred_image_to_json_post"),
        *prep("mask_post", "pred_image_to_json_mask_post"),
    )
def _generate_predicted_overlay_generic(selected_image_names, mapping_key):
    if not selected_image_names:
        tag = "pre" if mapping_key.endswith("_pre") else "post"
        cache_key = f"blurred_overlay_{tag}"

        if cache_key in SESSION:  # reuse cached blurred overlay
            return SESSION[cache_key]

        # generate blurred overlay only once
        if not SESSION.get("input_path") or not SESSION.get("line_json"):
            return None
        with open(SESSION["line_json"], "r") as f:
            line_data = json.load(f)
        if not line_data:
            return None

        fname = line_data[0]["image_name"]  # ✅ consistent with detection

        img_path = SESSION["input_path"] / fname
        base_img = Image.open(img_path).convert("RGB")
        blurred = base_img.filter(ImageFilter.GaussianBlur(radius=6))
        draw = ImageDraw.Draw(blurred)

        # draw axes
        for entry in line_data:
            if entry["image_name"] == fname:
                for x in sorted(entry["x_coordinates"]):
                    draw.line([(x, 0), (x, base_img.height)], fill="blue", width=2)
                break

        out_path = Path("outputs/reals/predicted_overlay") / f"{Path(fname).stem}_overlay_blurred_{tag}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        blurred.save(out_path)

        SESSION[cache_key] = str(out_path)  # cache path
        return str(out_path)

    mapping = SESSION.get(mapping_key, {})
    if not mapping:
        return None

    chosen_imgs = list(mapping.keys()) if "ALL" in selected_image_names \
        else [n for n in selected_image_names if n in mapping]
    if not chosen_imgs:
        return None

    # --- 🔎 Parse first JSON filename to extract id/suffix/tag ---
    first_json = Path(mapping[chosen_imgs[0]])
    stem = first_json.stem  # e.g. "image_1_crop_2_zdDU_pre"
    m = re.match(r'(.+_crop_\d+)_([^_]+)_(pre|mask|post|mask_post)$', stem)
    if not m:
        return None
    image_id = m.group(1)  # "image_1_crop_2"
    suffix = m.group(2)  # "SOXR"
    tag = m.group(3)  # "pre"

    base_id = image_id.split("_crop_")[0]

    with open(SESSION["line_json"], "r") as f:
        line_data = json.load(f)


    orig_entry = next((e for e in line_data if Path(e["image_name"]).stem == base_id), None)
    if not orig_entry:
        print(f"⚠️ Could not find base_id={base_id} in line_json")
        return None

    orig_fname = orig_entry["image_name"]

    base_img = Image.open(SESSION["input_path"] / orig_fname).convert("RGB")
    blurred = base_img.filter(ImageFilter.GaussianBlur(radius=6))
    output = blurred.copy()

    # load verticals / axes
    x_coords = []
    for entry in line_data:
        if entry.get("image_name") == orig_fname:
            x_coords = sorted(entry.get("x_coordinates", []))
            break
    if len(x_coords) < 2:
        out_path = Path("outputs/reals/predicted_overlay") / f"{image_id}_{suffix}_pred_overlay_{tag}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output.save(out_path)
        return str(out_path)

    H = base_img.height
    # detect crop index base (0 or 1)
    crop_indices_found = []
    for n in os.listdir(SESSION["cropped_dir"]):
        mm = re.search(r"_crop_(\d+)", n)
        if mm: crop_indices_found.append(int(mm.group(1)))
    base0 = 0 if crop_indices_found and min(crop_indices_found) == 0 else 1

    def crop_idx_to_box(crop_idx: int):
        i = crop_idx - base0
        if i < 0 or i >= len(x_coords) - 1:
            return None
        return (x_coords[i], 0, x_coords[i + 1], H)

    # axes
    draw = ImageDraw.Draw(output)
    for x in x_coords:
        draw.line([(x, 0), (x, H)], fill="blue", width=2)

    def svg_to_rgba(svg_path: Path, size_wh):
        try:
            buf = cairosvg.svg2png(url=str(svg_path), background_color="transparent")
            im = Image.open(io.BytesIO(buf)).convert("RGBA").resize(size_wh, Image.LANCZOS)
        except Exception:
            rgb = Image.open(svg_path).convert("RGB").resize(size_wh, Image.LANCZOS)
            im = rgb.convert("RGBA")
        rgb = im.convert("RGB")
        white = Image.new("RGB", size_wh, "white")
        diff = ImageChops.difference(rgb, white).convert("L")
        alpha = diff.point(lambda p: 255 if p > 10 else 0)
        im.putalpha(alpha)
        return im

    # group chosen images by crop index
    by_crop = {}
    for img_name in chosen_imgs:
        m = re.match(r'(.+_crop_(\d+))_([^_]+)_(pre|mask|post|mask_post)\.svg$', img_name)
        if not m:
            continue
        crop_idx = int(m.group(2))
        suff = m.group(3)
        by_crop.setdefault(crop_idx, []).append((img_name, suff))

    for crop_idx, items in by_crop.items():
        box = crop_idx_to_box(crop_idx)
        if not box:
            continue

        w, h = box[2] - box[0], box[3] - box[1]
        sharp_crop = base_img.crop(box).convert("RGBA")
        blurred_crop = output.crop(box).convert("RGBA")
        composited = blurred_crop.copy()

        for img_name, suff in items:
            json_path = Path(mapping[img_name])
            svg_path = json_path.with_suffix(".svg")
            if not svg_path.exists():
                continue
            overlay_rgba = svg_to_rgba(svg_path, (w, h))
            mask = overlay_rgba.split()[-1]
            composited = Image.composite(sharp_crop, composited, mask)
            composited.alpha_composite(overlay_rgba)

        output.paste(composited.convert("RGB"), box)

    # --- 🔑 Save with suffix and tag to keep variants separate ---
    out_path = Path("outputs/reals/predicted_overlay") / f"{image_id}_{suffix}_pred_overlay_{tag}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(out_path)
    return str(out_path)


def _generate_predicted_overlay_generic_m(selected_image_names, mapping_key):
    if not selected_image_names:
        tag = "pre" if mapping_key.endswith("_pre") else "post"
        cache_key = f"blurred_overlay_{tag}"

        if cache_key in SESSION:  # reuse cached blurred overlay
            return SESSION[cache_key]

        # generate blurred overlay only once
        if not SESSION.get("input_path") or not SESSION.get("line_json"):
            return None
        with open(SESSION["line_json"], "r") as f:
            line_data = json.load(f)
        if not line_data:
            return None

        fname = line_data[0]["image_name"]  # ✅ consistent with detection

        img_path = SESSION["input_path"] / fname
        base_img = Image.open(img_path).convert("RGB")
        blurred = base_img.filter(ImageFilter.GaussianBlur(radius=6))
        draw = ImageDraw.Draw(blurred)

        # draw axes
        with open(SESSION["line_json"], "r") as f:
            line_data = json.load(f)
        for entry in line_data:
            if entry["image_name"] == fname:
                for x in sorted(entry["x_coordinates"]):
                    draw.line([(x, 0), (x, base_img.height)], fill="blue", width=2)
                break

        out_path = Path("outputs/reals/predicted_overlay") / f"{Path(fname).stem}_overlay_blurred_{tag}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        blurred.save(out_path)

        SESSION[cache_key] = str(out_path)  # cache path
        return str(out_path)

    mapping = SESSION.get(mapping_key, {})
    if not mapping:
        return None

    chosen_imgs = list(mapping.keys()) if "ALL" in selected_image_names \
        else [n for n in selected_image_names if n in mapping]
    if not chosen_imgs:
        return None

    first_json = Path(mapping[chosen_imgs[0]])
    stem = first_json.stem  # e.g. "hjdhfkjfhh_crop_2_cat_3_post"
    parts = stem.rsplit("_crop_", 1)
    if len(parts) != 2:
        return None
    image_id = parts[0]  # ✅ full original filename stem

    # find exact filename from line_json
    with open(SESSION["line_json"], "r") as f:
        line_data = json.load(f)
    orig_entry = next((e for e in line_data if Path(e["image_name"]).stem == image_id), None)
    if not orig_entry:
        return None

    orig_fname = orig_entry["image_name"]  # ✅ exact original filename

    base_img = Image.open(SESSION["input_path"] / orig_fname).convert("RGB")
    blurred = base_img.filter(ImageFilter.GaussianBlur(radius=6))
    output = blurred.copy()

    # load verticals / axes
    with open(SESSION["line_json"], "r") as f:
        line_data = json.load(f)
    x_coords = []
    for entry in line_data:
        if entry.get("image_name") == orig_fname:
            x_coords = sorted(entry.get("x_coordinates", []))
            break
    if len(x_coords) < 2:
        out_path = Path("outputs/reals/predicted_overlay") / f"{image_id}_pred_overlay.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output.save(out_path)
        return str(out_path)

    H = base_img.height
    # detect crop index base (0 or 1)
    crop_indices_found = []
    for n in os.listdir(SESSION["cropped_dir"]):
        mm = re.search(r"_crop_(\d+)", n)
        if mm: crop_indices_found.append(int(mm.group(1)))
    base0 = 0 if crop_indices_found and min(crop_indices_found) == 0 else 1

    def crop_idx_to_box(crop_idx: int):
        i = crop_idx - base0
        if i < 0 or i >= len(x_coords) - 1:
            return None
        return (x_coords[i], 0, x_coords[i + 1], H)

    # axes
    draw = ImageDraw.Draw(output)
    for x in x_coords:
        draw.line([(x, 0), (x, H)], fill="blue", width=2)

    def svg_to_rgba(svg_path: Path, size_wh):
        try:
            import cairosvg, io
            buf = cairosvg.svg2png(url=str(svg_path), background_color="transparent")
            im = Image.open(io.BytesIO(buf)).convert("RGBA").resize(size_wh, Image.LANCZOS)
        except Exception:
            rgb = Image.open(svg_path).convert("RGB").resize(size_wh, Image.LANCZOS)
            im = rgb.convert("RGBA")
        rgb = im.convert("RGB")
        white = Image.new("RGB", size_wh, "white")
        diff = ImageChops.difference(rgb, white).convert("L")
        alpha = diff.point(lambda p: 255 if p > 10 else 0)
        im.putalpha(alpha)
        return im

    for img_name in chosen_imgs:
        json_path = Path(mapping[img_name])
        mm = re.search(r"_crop_(\d+)", json_path.name)
        if not mm:
            continue
        crop_idx = int(mm.group(1))
        box = crop_idx_to_box(crop_idx)
        if not box:
            continue

        svg_path = json_path.with_suffix(".svg")
        if not svg_path.exists():
            continue

        w, h = box[2] - box[0], box[3] - box[1]
        overlay_rgba = svg_to_rgba(svg_path, (w, h))
        mask = overlay_rgba.split()[-1]

        sharp_crop = base_img.crop(box).convert("RGBA")
        blurred_crop = output.crop(box).convert("RGBA")

        sharpened_under_lines = Image.composite(sharp_crop, blurred_crop, mask)
        composited = sharpened_under_lines.copy()
        composited.alpha_composite(overlay_rgba)

        output.paste(composited.convert("RGB"), box)

    # save per source to avoid overwriting
    tag = "pre" if mapping_key.endswith("_pre") else "post"
    out_path = Path("outputs/reals/predicted_overlay") / f"{image_id}_pred_overlay_{tag}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(out_path)
    return str(out_path)

def generate_predicted_overlay_pre(selected_image_names):
    return _generate_predicted_overlay_generic(selected_image_names, "pred_image_to_json_pre")

def generate_predicted_overlay_post(selected_image_names):
    return _generate_predicted_overlay_generic(selected_image_names, "pred_image_to_json_post")


def generate_predicted_overlay_mask(selected_image_names):
    return _generate_predicted_overlay_generic(selected_image_names, "pred_image_to_json_mask")

def generate_predicted_overlay_mask_post(selected_image_names):
    return _generate_predicted_overlay_generic(selected_image_names, "pred_image_to_json_mask_post")


def select_prediction_from_gallery_pre(evt: gr.SelectData):
    """
    Map pre-gallery click to its filename for single-selection.
    """
    mapping = SESSION.get("pred_image_to_json_pre", {})
    svg_files = list(mapping.keys())
    if 0 <= evt.index < len(svg_files):
        return [svg_files[evt.index]]
    return []

def select_prediction_from_gallery_post(evt: gr.SelectData):
    """
    Map post-gallery click to its filename for single-selection.
    """
    mapping = SESSION.get("pred_image_to_json_post", {})
    svg_files = list(mapping.keys())
    if 0 <= evt.index < len(svg_files):
        return [svg_files[evt.index]]
    return []

def select_prediction_from_gallery_mask(evt: gr.SelectData):
    """
    Map mask-gallery click to its filename for single-selection.
    """
    mapping = SESSION.get("pred_image_to_json_mask", {})
    svg_files = list(mapping.keys())
    if 0 <= evt.index < len(svg_files):
        return [svg_files[evt.index]]
    return []


def select_prediction_from_gallery_mask_post(evt: gr.SelectData):
    """
    Map mask+post-gallery click to its filename for single-selection.
    """
    mapping = SESSION.get("pred_image_to_json_mask_post", {})
    svg_files = list(mapping.keys())
    if 0 <= evt.index < len(svg_files):
        return [svg_files[evt.index]]
    return []


def save_predictions_as_npz(json_files, kind, out_root="outputs/reals/pred_npz"):
    out_dir = Path(out_root) / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)

        lines = np.array(data.get("lines", []), dtype=float).reshape(-1, 2, 2)
        scores = np.array(data.get("score", [1.0] * len(lines)), dtype=float)

        prefix = out_dir / Path(jf).stem
        np.savez_compressed(str(prefix) + ".npz", lines=lines, score=scores)
        saved.append(str(prefix) + ".npz")
    return saved


def toggle_prediction_sections(kinds):
    kinds = set(kinds or [])
    def vis(name): return gr.update(visible=(name in kinds))
    # order: pre, mask, post, mask_post
    return vis("pre"), vis("mask"), vis("post"), vis("mask_post")
