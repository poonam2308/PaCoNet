import os, json, cv2
import numpy as np
from pathlib import Path
from .session import SESSION, SESSION_LOG
from ..dhlp.dataset.gen_mask import generate_binary_mask
from ..dhlp.dataset.wireframe_test import save_heatmap
from ..dhlp.evaluation import  process_line_detection_arrays

from glob import glob

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
        coords = np.round(coords, 2)
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
    denoised_dir = SESSION.get("denoised_dir")
    lines_data = SESSION.get("separated_data_json")
    if not denoised_dir or not lines_data:
        return "Missing input data.", None

    with open(lines_data, "r") as f:
        linesdata = json.load(f)

    target_size = 224  # <<< keep everything at 224×224
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

        # Dynamically infer original width/height from coordinates
        max_x = coords[:, :, 0].max()
        max_y = coords[:, :, 1].max()
        orig_w = max_x if max_x > 0 else 1
        orig_h = max_y if max_y > 0 else 1

        # Scale coordinates into 224×224
        coords[:, :, 0] = coords[:, :, 0] / orig_w * target_size
        coords[:, :, 1] = coords[:, :, 1] / orig_h * target_size
        coords = np.round(coords, 2)
        # --- Prepare mask in 224×224 ---
        mask = generate_binary_mask(str(img_path))
        mask_resized = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

        # --- Load predictions ---
        pattern = f"outputs/reals/pred_npz/{kind}/{img_path.stem}_*.npz"
        pred_files = glob(pattern)
        if not pred_files:
            print("No preds for:", pattern)
            continue

        for pf in pred_files:
            pred_data = np.load(pf)
            predicted_lines = pred_data["lines"]

            # --- Normalize preds into 224×224 ---
            h_pred, w_pred = im.shape[:2]   # prediction is over denoised image size
            predicted_lines[:, :, 0] = predicted_lines[:, :, 0] / w_pred * target_size
            predicted_lines[:, :, 1] = predicted_lines[:, :, 1] / h_pred * target_size

            # --- Evaluate ---
            map_score = process_line_detection_arrays(coords, predicted_lines, mask_resized)
            total_map += map_score
            count += 1
            print(f"Processed {img_path.name} from {pf}: {map_score:.2f}")

    avg_map = total_map / count if count > 0 else 0
    return f"Average mAP ({kind}): {avg_map:.2f}", avg_map


