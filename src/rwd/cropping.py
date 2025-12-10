import cv2
import os
import json

def crop_images_single_dir(image_folder, json_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    with open(json_file, 'r') as f:
        data = json.load(f)
    for item in data:
        image_path = os.path.join(image_folder, item["image_name"])
        img = cv2.imread(os.path.join(image_folder, item["image_name"]))
        if img is None:
            print(f"DEBUG: Failed to load image: {image_path}")
            continue

        print(f"DEBUG: Successfully loaded image: {image_path} with shape {img.shape}")
        coords = sorted(item["x_coordinates"])
        print(f"DEBUG: X-coordinates for {item['image_name']}: {coords}")

        coords = sorted(item["x_coordinates"])
        for i in range(len(coords)-1):
            x1, x2 = coords[i], coords[i+1]
            if x2 - x1 <= 10:
                continue
            crop = img[:, x1:x2]
            out_path = os.path.join(output_folder, f"{os.path.splitext(item['image_name'])[0]}_crop_{i+1}.png")
            cv2.imwrite(out_path, crop)


def crop_images(image_folder, json_file, output_folder):
    # Main output folder (root)
    os.makedirs(output_folder, exist_ok=True)

    with open(json_file, 'r') as f:
        data = json.load(f)

    for item in data:
        image_name = item["image_name"]                  # e.g. "r261.png"
        image_base = os.path.splitext(image_name)[0]     # e.g. "r261"

        image_path = os.path.join(image_folder, image_name)
        img = cv2.imread(image_path)
        if img is None:
            print(f"DEBUG: Failed to load image: {image_path}")
            continue

        print(f"DEBUG: Successfully loaded image: {image_path} with shape {img.shape}")
        coords = sorted(item["x_coordinates"])
        print(f"DEBUG: X-coordinates for {image_name}: {coords}")

        # create a subfolder for this image, named by the image base name
        image_output_dir = os.path.join(output_folder, image_base)
        os.makedirs(image_output_dir, exist_ok=True)

        for i in range(len(coords) - 1):
            x1, x2 = coords[i], coords[i + 1]
            if x2 - x1 <= 10:
                continue

            crop = img[:, x1:x2]

            # ✅ include image_base in the crop filename
            crop_filename = f"{image_base}_crop_{i+1}.png"
            out_path = os.path.join(image_output_dir, crop_filename)

            cv2.imwrite(out_path, crop)
            print(f"Saved crop: {out_path}")

