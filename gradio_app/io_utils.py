from pathlib import Path
from PIL import Image

def save_uploaded_file(file_obj, tmp_dir: Path) -> str:
    ext = Path(file_obj).suffix.lower()
    new_name = Path(file_obj).stem + ".png" if ext != ".png" else Path(file_obj).name
    dest = tmp_dir / new_name
    img = Image.open(file_obj).convert("RGB")
    img.save(dest)
    return str(dest)


