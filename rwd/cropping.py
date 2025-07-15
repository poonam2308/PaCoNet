import cv2
import os
import json

def crop_images(image_folder, json_file, output_folder):
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
