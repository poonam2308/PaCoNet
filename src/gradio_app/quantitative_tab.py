import os, json, cv2
import numpy as np
from pathlib import Path
from .session import SESSION, SESSION_LOG
from ..dhlp.dataset.gen_mask import generate_binary_mask
from ..dhlp.dataset.wireframe_test import save_heatmap
from ..dhlp.evaluation import  process_line_detection_arrays


def generate_npz_from_denoised():
    denoised_dir = SESSION.get("denoised_dir")
    if not denoised_dir or not denoised_dir.exists():
        return [], "No denoised images found."

    # Change to look for separated_data_json
    lines_data = SESSION.get("separated_data_json")  # <--- Change this line
    if not lines_data or not Path(lines_data).exists():
        return [], "No separated data JSON found. Please run category separation first."  # <--- Update message

    with open(lines_data, "r") as f:
        linesdata = json.load(f)

    out_dir = Path("outputs/reals/npz_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for img_path in sorted(denoised_dir.glob("*.png")):
        im = cv2.imread(str(img_path))
        if im is None:
            continue

        # Match metadata entry by filename stem
        stem = img_path.stem.split("_crop_")[0]
        entry = next((e for e in linesdata if e["filename"].startswith(stem)), None)
        if not entry or "lines" not in entry:
            continue

        coords = np.array(entry["lines"], dtype=float).reshape(-1, 2, 2)

        # Scale coordinates to denoised image size
        orig_h, orig_w = entry.get("height"), entry.get("width")
        new_h, new_w = im.shape[:2]
        if orig_h and orig_w:
            sx, sy = new_w / orig_w, new_h / orig_h
            coords[:, :, 0] *= sx
            coords[:, :, 1] *= sy

        prefix = out_dir / img_path.stem

        # --- (1) Save wireframe npz ---
        save_heatmap(str(prefix), im, coords)
        results.append(str(prefix) + "_label.npz")

        # --- (2) Save mask npz ---
        mask = generate_binary_mask(str(img_path))
        mask_resized = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
        mask_npz_path = str(prefix) + "_mask_label.npz"
        np.savez_compressed(mask_npz_path, mask=mask_resized)
        results.append(mask_npz_path)

    SESSION_LOG["steps"].append("NPZ + Mask Conversion Completed")
    SESSION_LOG["results"]["npz_files"] = results

    return results, f"Conversion complete: {len(results)} files (wireframes + masks)."



def perform_quantitative_evaluation_in_memory(kind="post"):
    """
    In-memory evaluation (no intermediate NPZ).
    Uses arrays directly with process_line_detection_arrays.
    """
    denoised_dir = SESSION.get("denoised_dir")
    lines_data = SESSION.get("separated_data_json")
    if not denoised_dir or not lines_data:
        return "Missing input data.", None

    with open(lines_data, "r") as f:
        linesdata = json.load(f)

    total_map, count = 0, 0
    for img_path in sorted(denoised_dir.glob("*.png")):
        im = cv2.imread(str(img_path))
        if im is None:
            continue

        stem = img_path.stem.split("_crop_")[0]
        entry = next((e for e in linesdata if e["filename"].startswith(stem)), None)
        if not entry or "lines" not in entry:
            continue

        coords = np.array(entry["lines"], dtype=float).reshape(-1, 2, 2)

        orig_h, orig_w = entry.get("height"), entry.get("width")
        new_h, new_w = im.shape[:2]
        if orig_h and orig_w:
            sx, sy = new_w / orig_w, new_h / orig_h
            coords[:, :, 0] *= sx
            coords[:, :, 1] *= sy

        mask = generate_binary_mask(str(img_path))
        mask_resized = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)

        pred_path = Path(f"outputs/reals/pred_npz/{kind}/{img_path.stem}.npz")
        if not pred_path.exists():
            continue
        pred_data = np.load(pred_path)
        print(pred_data)
        predicted_lines = pred_data["lines"]


        map_score = process_line_detection_arrays(coords, predicted_lines, mask_resized)
        print(map_score)
        total_map += map_score
        count += 1
        print(f"Processed {img_path.name}: {map_score:.2f}")

    avg_map = total_map / count if count > 0 else 0
    return f"Average mAP ({kind}): {avg_map:.2f}", avg_map
