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
