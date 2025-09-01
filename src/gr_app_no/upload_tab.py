import os
import json
import shutil
import zipfile
import tempfile

def process_upload(image, annotation, folder_files):
    if folder_files:
        if len(folder_files) == 1 and folder_files[0].endswith(".zip"):
            tmpdir = tempfile.mkdtemp()
            with zipfile.ZipFile(folder_files[0], 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            files = os.listdir(tmpdir)
            return f"Uploaded ZIP with {len(files)} extracted files.", tmpdir
        else:
            tmpdir = tempfile.mkdtemp()
            for f in folder_files:
                shutil.copy(f, tmpdir)
            return f"Uploaded {len(folder_files)} files (treated as folder).", tmpdir

    if image and annotation:
        try:
            with open(annotation, "r") as f:
                data = json.load(f)
            tmpdir = tempfile.mkdtemp()
            shutil.copy(image, tmpdir)
            shutil.copy(annotation, tmpdir)
            return f"Uploaded single image + annotation with {len(data)} keys.", tmpdir
        except Exception as e:
            return f"Error reading annotation: {e}", None

    return "Please upload an image + annotation, a zip, or multiple files.", None
