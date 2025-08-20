from pathlib import Path
import json

SESSION = {
    "input_path": None,
    "line_output_dir": None,
    "line_json": None,
    "cropped_dir": None,
    "selected_dir": None,
    "separated_dir": None,
    "denoised_dir": None,
    "pred_image_to_json_pre": {},
    "pred_image_to_json_post": {}
}

SESSION_LOG = {
    "inputs": {},
    "steps": [],
    "results": {}
}


def save_session_log() -> str:
    log_path = Path("outputs/reals/session_log.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(SESSION_LOG, f, indent=2)
    return str(log_path)

# from pathlib import Path
# import json
#
# SESSION = {
#     "images": {},        # 👈 each key = image filename
#     "active_image": None # 👈 current image user is viewing
# }
#
# SESSION_LOG = {
#     "inputs": {},
#     "steps": [],
#     "results": {}
# }
#
#
# def init_image_session(img_name: str, img_path: Path, json_path: Path, output_root: Path):
#     """
#     Initialize SESSION entries for one image+json pair.
#     """
#     SESSION["images"][img_name] = {
#         "input_path": img_path,  # actual image path
#         "metadata_json": json_path,  # original metadata file or generated JSON
#         "cropped_dir": output_root / "cropped" / Path(img_name).stem,
#         "selected_dir": output_root / "selected" / Path(img_name).stem,
#         "separated_dir": output_root / "separated" / Path(img_name).stem,
#         "denoised_dir": output_root / "denoised" / Path(img_name).stem
#     }
#
#     if SESSION["active_image"] is None:
#         SESSION["active_image"] = img_name
#
#
#
# def save_session_log() -> str:
#     log_path = Path("outputs/reals/session_log.json")
#     log_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(log_path, "w") as f:
#         json.dump(SESSION_LOG, f, indent=2)
#     return str(log_path)

