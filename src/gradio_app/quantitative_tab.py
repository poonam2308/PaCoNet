import os, json, cv2
import numpy as np
from pathlib import Path
from .session import SESSION, SESSION_LOG
from ..dhlp.dataset.gen_mask import generate_binary_mask
from ..dhlp.dataset.wireframe_test import save_heatmap
from ..dhlp.eval_sAP_ps import process_multiple_files

def generate_npz_from_denoised():
    denoised_dir = SESSION.get("denoised_dir")
    if not denoised_dir or not denoised_dir.exists():
        return [], "No denoised images found."

    meta_path = SESSION.get("metadata_json")
    if not meta_path or not Path(meta_path).exists():
        return [], "No metadata JSON found."

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    out_dir = Path("outputs/reals/npz_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for img_path in sorted(denoised_dir.glob("*.png")):
        im = cv2.imread(str(img_path))
        if im is None:
            continue

        # Match metadata entry by filename stem
        stem = img_path.stem.split("_crop_")[0]
        entry = next((e for e in metadata if e["image_name"].startswith(stem)), None)
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


def perform_quantitative_evaluation(kind="post"):
    # Step 1: Regenerate NPZs
    results, status = generate_npz_from_denoised()
    if not results:
        return status, None

    # Step 2: Define directories
    gt_dir = "outputs/reals/npz_data"
    mask_dir = "outputs/reals/npz_data"
    pred_dir = f"outputs/reals/pred_npz/{kind}"  # choose specific variant

    # Step 3: Run evaluation
    avg_map = process_multiple_files(gt_dir, pred_dir, mask_dir)
    return f"{status}\nAverage mAP ({kind}): {avg_map:.2f}", avg_map

