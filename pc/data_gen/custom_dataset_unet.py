import json
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from scipy.spatial.distance import euclidean

# Load JSON files
def load_json(json_path):
    with open(json_path, "r") as file:
        return json.load(file)


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
