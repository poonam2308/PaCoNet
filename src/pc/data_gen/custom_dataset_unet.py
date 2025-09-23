import json
import os
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import numpy as np
import torchvision.transforms as transforms
from scipy.spatial.distance import euclidean

# Load JSON files
def load_json(json_path):
    with open(json_path, "r") as file:
        return json.load(file)


def extract_base_name(filename: str):
    """
    Extracts a canonical base name from a filename by keeping only
    the image ID (e.g., image_1) and crop ID (e.g., crop_1).
    Example:
        image_1_crop_1_cat.png -> image_1_crop_1
        image_1_cat_crop_1.png -> image_1_crop_1
        image_1_crop_1.png     -> image_1_crop_1
    """
    name = filename.replace(".png", "").replace(".jpg", "").replace(".jpeg", "")
    parts = name.split("_")

    image_id = None
    crop_id = None

    for i, part in enumerate(parts):
        # detect image number (e.g., "image_1")
        if part.startswith("image"):
            if i + 1 < len(parts) and parts[i+1].isdigit():
                image_id = f"{part}_{parts[i+1]}"
            else:
                image_id = part
        # detect crop number (e.g., "crop_1")
        if part.startswith("crop"):
            if i + 1 < len(parts) and parts[i+1].isdigit():
                crop_id = f"{part}_{parts[i+1]}"
            else:
                crop_id = part

    base = []
    if image_id: base.append(image_id)
    if crop_id: base.append(crop_id)
    return "_".join(base)

class CustomDatasetUnetSD(Dataset):
    def __init__(self, input_json=None, input_dir=None,
                 ground_truth_json=None, ground_truth_dir=None,
                 transform=None, channel_mode ="RGB", hsv_tolerance=0.1, remove_background=False):

        # Load input data (from JSON or directory)
        if input_json:
            self.input_data = load_json(input_json)
            self.input_filenames = [item["filename"] for item in self.input_data]
        else:
            # Build minimal records so match_pairs can iterate safely
            self.input_filenames = [f for f in os.listdir(input_dir)
                                    if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            self.input_data = [{"filename": f} for f in self.input_filenames]

        self.input_dir = input_dir

        # Load GT (if provided)
        self.ground_truth_data = load_json(ground_truth_json) if ground_truth_json else None
        self.ground_truth_dir = ground_truth_dir

        self.transform = transform
        self.hsv_tolerance = hsv_tolerance
        self.remove_background = remove_background
        self.channel_mode = channel_mode

        self.pairs = self.match_pairs()

    def match_pairs(self):
        pairs = []

        # Index ground truths by base name if JSON is given
        gt_dict = {}
        if self.ground_truth_data:
            for gt_item in self.ground_truth_data:
                base = extract_base_name(gt_item["filename"])
                gt_dict.setdefault(base, []).append(gt_item)

        # If no GT JSON, but a GT directory exists, build a filename index by base name
        gt_dir_index = {}
        if not self.ground_truth_data and self.ground_truth_dir and os.path.isdir(self.ground_truth_dir):
            for f in os.listdir(self.ground_truth_dir):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    base = extract_base_name(f)
                    gt_dir_index.setdefault(base, []).append(f)

        for input_item in self.input_data:
            input_filename = input_item["filename"]
            base_name_input = extract_base_name(input_filename)

            input_hsv = input_item.get("color_hsv", None)

            best_match = None
            best_hsv_distance = float("inf")

            # Prefer JSON GT if available
            candidate_gts = []
            if base_name_input in gt_dict:
                candidate_gts = gt_dict[base_name_input]
            elif base_name_input in gt_dir_index:
                # Convert directory filenames into minimal records to unify logic
                candidate_gts = [{"filename": f, "color_hsv": None} for f in gt_dir_index[base_name_input]]

            # Try HSV-based selection if both sides have HSV; otherwise fall back to first candidate
            if candidate_gts:
                if input_hsv is not None and any("color_hsv" in g and g["color_hsv"] is not None for g in candidate_gts):
                    from scipy.spatial.distance import euclidean
                    for gt_item in candidate_gts:
                        gt_hsv = gt_item.get("color_hsv")
                        if gt_hsv is None:
                            continue  # skip HSV-less candidates in HSV mode
                        hsv_distance = euclidean(
                            [input_hsv['h'], input_hsv['s'], input_hsv['v']],
                            [gt_hsv['h'], gt_hsv['s'], gt_hsv['v']]
                        )
                        if hsv_distance < self.hsv_tolerance and hsv_distance < best_hsv_distance:
                            best_match = gt_item["filename"]
                            best_hsv_distance = hsv_distance
                    # If none within tolerance, still take the closest if any were computed
                    if best_match is None and best_hsv_distance < float("inf"):
                        # choose the candidate with min distance (already tracked)
                        # Nothing to do: best_match stays None if we never set it; handle below
                        pass
                # Fallback: no HSV available or no within-tolerance -> take first candidate
                if best_match is None:
                    best_match = candidate_gts[0]["filename"]

            # If still nothing found, pair with None (your __getitem__ checks this)
            if best_match:
                pairs.append((input_filename, best_match))
            else:
                print(f"Warning: No match found for {input_filename}. Pairing with None.")
                pairs.append((input_filename, None))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def remove_bg(self, image):
        img_array = np.array(image)
        mask = np.any(img_array < 250, axis=-1)
        white_bg = np.ones_like(img_array) * 255
        white_bg[mask] = img_array[mask]
        return Image.fromarray(white_bg.astype(np.uint8))

    def __getitem__(self, idx):
        input_filename, gt_filename = self.pairs[idx]
        input_path = os.path.join(self.input_dir, input_filename)
        input_image = Image.open(input_path).convert(self.channel_mode)

        gt_image = None
        if gt_filename:
            gt_path = os.path.join(self.ground_truth_dir, gt_filename)
            gt_image = Image.open(gt_path).convert(self.channel_mode)

        if self.remove_background:
            input_image = self.remove_bg(input_image)
            if gt_image:
                gt_image = self.remove_bg(gt_image)

        if self.transform:
            input_image = self.transform(input_image)
            if gt_image:
                gt_image = self.transform(gt_image)

        return input_image, gt_image

class CustomDatasetUnetSD_useless(Dataset):
    def __init__(self, input_json=None, input_dir=None,
                 ground_truth_json=None, ground_truth_dir=None,
                 transform=None, hsv_tolerance=0.1, remove_background=False):

        # Load input data (from JSON or directory)
        if input_json:
            self.input_data = load_json(input_json)
            self.input_filenames = [item["filename"] for item in self.input_data]
        else:
            self.input_data = None
            self.input_filenames = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        self.input_dir = input_dir
        self.ground_truth_data = load_json(ground_truth_json) if ground_truth_json else None
        self.ground_truth_dir = ground_truth_dir
        self.transform = transform
        self.hsv_tolerance = hsv_tolerance
        self.remove_background = remove_background

        self.pairs = self.match_pairs()

    def match_pairs(self):
        pairs = []

        # Pre-index ground truths by base name
        gt_dict = {}
        if self.ground_truth_data:
            for gt_item in self.ground_truth_data:
                base = extract_base_name(gt_item["filename"])
                if base not in gt_dict:
                    gt_dict[base] = []
                gt_dict[base].append(gt_item)

        for input_item in self.input_data:
            input_filename = input_item["filename"]
            input_hsv = input_item["color_hsv"]
            base_name_input = extract_base_name(input_filename)

            best_match = None
            best_hsv_distance = float("inf")

            # Only compare against ground truths with the same base name
            if base_name_input in gt_dict:
                for gt_item in gt_dict[base_name_input]:
                    gt_hsv = gt_item["color_hsv"]
                    hsv_distance = euclidean(
                        [input_hsv['h'], input_hsv['s'], input_hsv['v']],
                        [gt_hsv['h'], gt_hsv['s'], gt_hsv['v']]
                    )
                    if hsv_distance < self.hsv_tolerance and hsv_distance < best_hsv_distance:
                        best_match = gt_item["filename"]
                        best_hsv_distance = hsv_distance

            if best_match:
                pairs.append((input_filename, best_match))
            else:
                print(f"Warning: No match found for {input_filename}. Skipping.")

        return pairs

    def __len__(self):
        return len(self.pairs)

    def remove_bg(self, image):
        """Replace non-white background with pure white"""
        img_array = np.array(image)
        # Mask for non-white pixels
        mask = np.any(img_array < 250, axis=-1)
        white_bg = np.ones_like(img_array) * 255
        white_bg[mask] = img_array[mask]
        return Image.fromarray(white_bg.astype(np.uint8))

    def __getitem__(self, idx):
        input_filename, gt_filename = self.pairs[idx]
        input_path = os.path.join(self.input_dir, input_filename)
        input_image = Image.open(input_path).convert("RGB")

        gt_image = None
        if gt_filename:
            gt_path = os.path.join(self.ground_truth_dir, gt_filename)
            gt_image = Image.open(gt_path).convert("RGB")

        # Background removal if flag is set
        if self.remove_background:
            input_image = self.remove_bg(input_image)
            if gt_image:
                gt_image = self.remove_bg(gt_image)

        if self.transform:
            input_image = self.transform(input_image)
            if gt_image:
                gt_image = self.transform(gt_image)
        return input_image, gt_image


class CustomHSVMatchingDatasetSD(Dataset):
    def __init__(self, input_json, ground_truth_json, input_dir, ground_truth_dir, transform=None, hsv_tolerance=0.1):
        self.input_data = load_json(input_json)
        self.ground_truth_data = load_json(ground_truth_json)
        self.input_dir = input_dir
        self.ground_truth_dir = ground_truth_dir
        self.transform = transform
        self.hsv_tolerance = hsv_tolerance
        self.pairs = self.match_pairs()

    def match_pairs(self):
        pairs = []
        for input_item in self.input_data:
            input_filename = input_item["filename"]
            input_hsv = input_item["color_hsv"]

            best_match = None
            best_hsv_distance = float('inf')

            for gt_item in self.ground_truth_data:
                gt_filename = gt_item["filename"]
                gt_hsv = gt_item["color_hsv"]

                # Extract base names for partial matching
                base_name_input = "_".join(input_filename.split('_')[:4])# e.g., image_1_crop_1
                # base_name_gt = "_".join(gt_filename.split('_')[:2])  # e.g., image_1
                base_name_gt = "_".join(gt_filename.split('_')[0:2] +
                                        gt_filename.split('_')[-2:])  # e.g., image_1_crop_1
                if base_name_gt.endswith('.png'):
                    base_name_gt = base_name_gt[:-4]

                if base_name_input.startswith(base_name_gt):  # Refined partial match
                    # Calculate HSV distance
                    hsv_distance = euclidean(
                        [input_hsv['h'], input_hsv['s'], input_hsv['v']],
                        [gt_hsv['h'], gt_hsv['s'], gt_hsv['v']]
                    )

                    if hsv_distance < self.hsv_tolerance and hsv_distance < best_hsv_distance:
                        best_match = gt_filename

                        best_hsv_distance = hsv_distance

            if best_match:
                pairs.append((input_filename, best_match))
            else:
                # Skip unmatched files and log a warning
                print(f"Warning: No match found for {input_filename}. Skipping.")

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_filename, gt_filename = self.pairs[idx]
        input_path = os.path.join(self.input_dir, input_filename)
        gt_path = os.path.join(self.ground_truth_dir, gt_filename)

        input_image = Image.open(input_path).convert("RGB")
        gt_image = Image.open(gt_path).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)
            gt_image = self.transform(gt_image)

        return input_image, gt_image, input_filename

class CustomHSVMatchingDataset(Dataset):
    def __init__(self, input_json, ground_truth_json, input_dir, ground_truth_dir, transform=None, hsv_tolerance=0.1):
        self.input_data = load_json(input_json)
        self.ground_truth_data = load_json(ground_truth_json)
        self.input_dir = input_dir
        self.ground_truth_dir = ground_truth_dir
        self.transform = transform
        self.hsv_tolerance = hsv_tolerance
        self.pairs = self.match_pairs()

    def match_pairs(self):
        pairs = []
        for input_item in self.input_data:
            input_filename = input_item["filename"]
            input_hsv = input_item["color_hsv"]

            best_match = None
            best_hsv_distance = float('inf')

            for gt_item in self.ground_truth_data:
                gt_filename = gt_item["filename"]
                gt_hsv = gt_item["color_hsv"]

                # Extract base names for partial matching
                base_name_input = "_".join(input_filename.split('_')[:4])# e.g., image_1_crop_1
                # base_name_gt = "_".join(gt_filename.split('_')[:2])  # e.g., image_1
                base_name_gt = "_".join(gt_filename.split('_')[0:2] +
                                        gt_filename.split('_')[-2:])  # e.g., image_1_crop_1
                if base_name_gt.endswith('.png'):
                    base_name_gt = base_name_gt[:-4]

                if base_name_input.startswith(base_name_gt):  # Refined partial match
                    # Calculate HSV distance
                    hsv_distance = euclidean(
                        [input_hsv['h'], input_hsv['s'], input_hsv['v']],
                        [gt_hsv['h'], gt_hsv['s'], gt_hsv['v']]
                    )

                    if hsv_distance < self.hsv_tolerance and hsv_distance < best_hsv_distance:
                        best_match = gt_filename

                        best_hsv_distance = hsv_distance

            if best_match:
                pairs.append((input_filename, best_match))
            else:
                # Skip unmatched files and log a warning
                print(f"Warning: No match found for {input_filename}. Skipping.")

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_filename, gt_filename = self.pairs[idx]
        input_path = os.path.join(self.input_dir, input_filename)
        gt_path = os.path.join(self.ground_truth_dir, gt_filename)

        input_image = Image.open(input_path).convert("RGB")
        gt_image = Image.open(gt_path).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)
            gt_image = self.transform(gt_image)

        return input_image, gt_image

class CustomTestDatasetSD1(Dataset):
    def __init__(self, input_json, input_dir, transform=None):
        self.input_data = load_json(input_json)  # Load test image metadata
        self.input_dir = input_dir
        self.transform = transform

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_filename = self.input_data[idx]["filename"]
        input_path = os.path.join(self.input_dir, input_filename)

        input_image = Image.open(input_path).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)

        return input_image, input_filename  # No ground truth, so returning only input image and filename

class CustomTestDatasetSD(Dataset):
    def __init__(self, input_dir, transform=None):
        self.input_dir = input_dir
        self.transform = transform

        # Get list of all image files in the directory
        self.image_files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        input_filename = self.image_files[idx]
        input_path = os.path.join(self.input_dir, input_filename)

        try:
            input_image = Image.open(input_path).convert("RGB")

            if self.transform:
                input_image = self.transform(input_image)

            return input_image, input_filename  # Returning only image and filename
        except Exception as e:
            print(f"Error loading image {input_filename}: {e}")
            return None  # Return None to indicate an issue


class CustomHSVMatchingDatasetSD1(Dataset):
    def __init__(self, input_json, ground_truth_json, input_dir, ground_truth_dir, transform=None, hsv_tolerance=0.1):
        self.input_data = load_json(input_json)
        self.ground_truth_data = load_json(ground_truth_json)
        self.input_dir = input_dir
        self.ground_truth_dir = ground_truth_dir
        self.transform = transform
        self.hsv_tolerance = hsv_tolerance
        self.pairs = self.match_pairs()

    def match_pairs(self):
        pairs = []

        # Group ground truth images by their base name
        gt_groups = {}
        for gt_item in self.ground_truth_data:
            gt_filename = gt_item["filename"]
            base_name_gt = "_".join(gt_filename.split('_')[0:2] + gt_filename.split('_')[-2:]).replace('.png', '')

            if base_name_gt not in gt_groups:
                gt_groups[base_name_gt] = []

            gt_groups[base_name_gt].append(gt_item)

        for input_item in self.input_data:
            input_filename = input_item["filename"]
            input_hsv = input_item["color_hsv"]
            base_name_input = "_".join(input_filename.split('_')[:4])  # e.g., image_1_crop_1

            best_match = None
            best_hsv_distance = float('inf')
            closest_match = None
            closest_hsv_distance = float('inf')

            # Check if there's a matching ground truth group
            if base_name_input in gt_groups:
                gt_items = gt_groups[base_name_input]

                # Get the lowest and highest HSV values in this ground truth group
                h_values = [gt["color_hsv"]["h"] for gt in gt_items]
                min_h = min(h_values) if h_values else None
                max_h = max(h_values) if h_values else None

                # **Apply forced adjustment BEFORE finding closest match**
                adjusted_h = input_hsv["h"]
                if adjusted_h <= 0.07 and min_h is not None:
                    adjusted_h = min_h
                elif adjusted_h > 0.9 and max_h is not None:
                    adjusted_h = max_h

                for gt_item in gt_items:
                    gt_filename = gt_item["filename"]
                    gt_hsv = gt_item["color_hsv"]

                    # Compute HSV distance **after** adjusting h value
                    hsv_distance = euclidean(
                        [adjusted_h, input_hsv['s'], input_hsv['v']],
                        [gt_hsv['h'], gt_hsv['s'], gt_hsv['v']]
                    )

                    # Select the best match within tolerance
                    if hsv_distance < self.hsv_tolerance and hsv_distance < best_hsv_distance:
                        best_match = gt_filename
                        best_hsv_distance = hsv_distance

                    # Track the closest match, even outside tolerance
                    if hsv_distance < closest_hsv_distance:
                        closest_match = gt_filename
                        closest_hsv_distance = hsv_distance

            # **Assign best match within tolerance, otherwise take the closest match**
            if best_match:
                pairs.append((input_filename, best_match))
            elif closest_match:
                pairs.append((input_filename, closest_match))
                print(
                    f"Warning: No match within tolerance for {input_filename}. Assigned closest match: {closest_match}")
            else:
                print(f"Warning: No match found for {input_filename}. Skipping.")

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_filename, gt_filename = self.pairs[idx]
        input_path = os.path.join(self.input_dir, input_filename)
        gt_path = os.path.join(self.ground_truth_dir, gt_filename)

        input_image = Image.open(input_path).convert("RGB")
        gt_image = Image.open(gt_path).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)
            gt_image = self.transform(gt_image)

        return input_image, gt_image

        # return input_image, gt_image, input_filename

# this dataset also includes the mask of the ground truth crops
class CustomHSVMatchingDatasetSDMask(Dataset):
    def __init__(self, input_json, ground_truth_json, input_dir, ground_truth_dir,
                 mask_dir, transform=None, hsv_tolerance=0.1):
        self.input_data = load_json(input_json)
        self.ground_truth_data = load_json(ground_truth_json)
        self.input_dir = input_dir
        self.ground_truth_dir = ground_truth_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.hsv_tolerance = hsv_tolerance
        self.pairs = self.match_pairs()

    def match_pairs(self):
        pairs = []

        # Group ground truth images by their base name
        gt_groups = {}
        for gt_item in self.ground_truth_data:
            gt_filename = gt_item["filename"]
            base_name_gt = "_".join(gt_filename.split('_')[0:2] + gt_filename.split('_')[-2:]).replace('.png', '')

            if base_name_gt not in gt_groups:
                gt_groups[base_name_gt] = []

            gt_groups[base_name_gt].append(gt_item)

        for input_item in self.input_data:
            input_filename = input_item["filename"]
            input_hsv = input_item["color_hsv"]
            base_name_input = "_".join(input_filename.split('_')[:4])  # e.g., image_1_crop_1

            best_match = None
            best_hsv_distance = float('inf')
            closest_match = None
            closest_hsv_distance = float('inf')

            # Check if there's a matching ground truth group
            if base_name_input in gt_groups:
                gt_items = gt_groups[base_name_input]

                # Get the lowest and highest HSV values in this ground truth group
                h_values = [gt["color_hsv"]["h"] for gt in gt_items]
                min_h = min(h_values) if h_values else None
                max_h = max(h_values) if h_values else None

                # **Apply forced adjustment BEFORE finding closest match**
                adjusted_h = input_hsv["h"]
                if adjusted_h <= 0.07 and min_h is not None:
                    adjusted_h = min_h
                elif adjusted_h > 0.9 and max_h is not None:
                    adjusted_h = max_h

                for gt_item in gt_items:
                    gt_filename = gt_item["filename"]
                    gt_hsv = gt_item["color_hsv"]

                    # Compute HSV distance **after** adjusting h value
                    hsv_distance = euclidean(
                        [adjusted_h, input_hsv['s'], input_hsv['v']],
                        [gt_hsv['h'], gt_hsv['s'], gt_hsv['v']]
                    )

                    # Select the best match within tolerance
                    if hsv_distance < self.hsv_tolerance and hsv_distance < best_hsv_distance:
                        best_match = gt_filename
                        best_hsv_distance = hsv_distance

                    # Track the closest match, even outside tolerance
                    if hsv_distance < closest_hsv_distance:
                        closest_match = gt_filename
                        closest_hsv_distance = hsv_distance

            # **Assign best match within tolerance, otherwise take the closest match**
            if best_match:
                pairs.append((input_filename, best_match))
            elif closest_match:
                pairs.append((input_filename, closest_match))
                print(
                    f"Warning: No match within tolerance for {input_filename}. Assigned closest match: {closest_match}")
            else:
                print(f"Warning: No match found for {input_filename}. Skipping.")

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_filename, gt_filename = self.pairs[idx]
        input_path = os.path.join(self.input_dir, input_filename)
        gt_path = os.path.join(self.ground_truth_dir, gt_filename)
        mask_path = os.path.join(self.mask_dir, input_filename)  # Assuming masks match GT filenames

        input_image = Image.open(input_path).convert("RGB")
        gt_image = Image.open(gt_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale

        if self.transform:
            input_image = self.transform(input_image)
            gt_image = self.transform(gt_image)

            # Apply only resizing and ToTensor to mask (no normalization)
            basic_mask_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            mask = basic_mask_transform(mask)

        # Binarize the mask: values should be 0 or 1
        mask = (mask > 0.5).float()

        return input_image, gt_image, mask
