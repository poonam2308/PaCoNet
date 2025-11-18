import json
import os
from collections import defaultdict

import re
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import numpy as np
import torchvision.transforms as transforms
from scipy.spatial.distance import euclidean

from src.pc.utils import load_json, parse_key


class CustomDatasetUnetSD(Dataset):
    def __init__(self, input_json=None, input_dir=None,
                 ground_truth_json=None, ground_truth_dir=None,
                 transform=None, channel_mode ="RGB", hsv_tolerance=0.1, remove_background=False,
                 bg_border=5,  # pixels sampled from each border to estimate bg color
                 bg_dist_thresh=45,  # RGB distance to bg color that counts as background
                 bg_soften=10,  # soft band around threshold for anti-aliased edges
                 v_black_thresh=0.18,  # HSV 'Value' below this → dark bg
                 v_white_thresh=0.80,  # HSV 'Value' above this + low sat → white/gray bg
                 s_low_thresh=0.18,
                 binarize=False, binarize_method="otsu", binarize_threshold=128):

        if input_json:
            self.input_data = load_json(input_json)
            self.input_filenames = [item["filename"] for item in self.input_data]
        else:
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

        self.binarize = binarize
        self.binarize_method = binarize_method
        self.binarize_threshold = binarize_threshold

        self.bg_border = bg_border
        self.bg_dist_thresh = bg_dist_thresh
        self.bg_soften = bg_soften
        self.v_black_thresh = v_black_thresh
        self.v_white_thresh = v_white_thresh
        self.s_low_thresh = s_low_thresh


        self.pairs = self.match_pairs()


    def parse_key(filename: str):
        """
        Return a normalized key for pairing:
            (image_id:int, crop_id:int, code:str|None)
        Accepts both 'image_3_crop_2_UTnXwc.png' and 'image_3_UTnXwc_crop_2.png'.
        """
        stem = os.path.splitext(os.path.basename(filename))[0]
        parts = stem.split('_')

        image_id, crop_id, code = None, None, None

        # find image id and crop id wherever they appear
        for i, p in enumerate(parts):
            if p == 'image' and i + 1 < len(parts) and parts[i + 1].isdigit():
                image_id = int(parts[i + 1])
            if p == 'crop' and i + 1 < len(parts) and parts[i + 1].isdigit():
                crop_id = int(parts[i + 1])

        # find an alphanumeric token that is not 'image'/'crop' or a pure number
        # (your random tag like UTnXwc, w06Cr, etc.)
        for p in parts:
            if p not in {'image', 'crop'} and not p.isdigit():
                code = p
                break

        return (image_id, crop_id, code)

    def match_pairs(self):
        """
        Deterministic matching:
          1) Try exact key (image_id, crop_id, code).
          2) If not found, try (image_id, crop_id) ignoring code.
          3) If multiple candidates remain, pick the one with the smallest HSV distance when available;
             otherwise pick lexicographically (stable).
        """
        # --- index GTs
        gt_by_full = defaultdict(list)  # (img, crop, code)
        gt_by_base = defaultdict(list)  # (img, crop)
        gt_items = []

        if self.ground_truth_data:
            for gt_item in self.ground_truth_data:
                k = parse_key(gt_item["filename"])
                gt_items.append((gt_item["filename"], gt_item.get("color_hsv")))
                gt_by_full[k].append(gt_item)
                gt_by_base[(k[0], k[1])].append(gt_item)
        else:
            for f in os.listdir(self.ground_truth_dir):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    k = parse_key(f)
                    gt_item = {"filename": f, "color_hsv": None}
                    gt_items.append((f, None))
                    gt_by_full[k].append(gt_item)
                    gt_by_base[(k[0], k[1])].append(gt_item)

        pairs = []
        for input_item in self.input_data:
            in_name = input_item["filename"]
            in_hsv = input_item.get("color_hsv")
            ik = parse_key(in_name)

            candidates = []
            # 1) exact key
            if ik in gt_by_full:
                candidates = gt_by_full[ik]
            # 2) fall back to (image, crop)
            elif (ik[0], ik[1]) in gt_by_base:
                candidates = gt_by_base[(ik[0], ik[1])]

            chosen = None
            if candidates:
                # if HSV present on both sides, choose smallest distance
                if in_hsv is not None and any(c.get("color_hsv") for c in candidates):
                    best_d, best = float("inf"), None
                    for c in candidates:
                        gh = c.get("color_hsv")
                        if gh is None:
                            continue
                        d = ((in_hsv['h'] - gh['h']) ** 2 + (in_hsv['s'] - gh['s']) ** 2 + (
                                    in_hsv['v'] - gh['v']) ** 2) ** 0.5
                        if d < best_d:
                            best_d, best = d, c
                    chosen = best if best is not None else sorted(candidates, key=lambda x: x["filename"])[0]
                else:
                    # deterministic fallback
                    chosen = sorted(candidates, key=lambda x: x["filename"])[0]

            if chosen is None:
                print(f"Warning: No GT found for {in_name} (parsed key={ik}).")
                continue

            pairs.append((in_name, chosen["filename"]))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def remove_bg(self, image):
        """
        Make a mostly-uniform background pure white (black, white, gray, tan, etc.)
        while keeping colored strokes unchanged. Uses:
          1) border sampling to estimate bg color,
          2) soft RGB distance threshold to that color,
          3) HSV helpers for very dark or very bright low-sat bgs.
        """
        img = image.convert("RGB")
        arr_u8 = np.asarray(img).astype(np.uint8)
        h, w, _ = arr_u8.shape
        arr = arr_u8.astype(np.float32)

        # 1) estimate background color from a thin border frame
        b = max(1, int(self.bg_border))
        strips = [arr[:b, :, :], arr[-b:, :, :], arr[:, :b, :], arr[:, -b:, :]]
        frame = np.concatenate([s.reshape(-1, 3) for s in strips], axis=0)
        bg_color = np.median(frame, axis=0)  # robust median RGB

        # 2) soft distance-to-bg mask (1=bg, 0=fg)
        diff = arr - bg_color[None, None, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))  # 0..~441
        low = float(self.bg_dist_thresh)
        high = low + max(1e-6, float(self.bg_soften))
        alpha_bg1 = np.clip(
            (dist <= low) * 1.0 + ((high - dist) / (high - low)) * ((dist > low) & (dist < high)),
            0.0, 1.0
        )

        # 3) HSV helpers (dark and bright-gray bgs)
        norm = arr / 255.0
        maxc = norm.max(axis=-1)
        minc = norm.min(axis=-1)
        v = maxc
        with np.errstate(divide='ignore', invalid='ignore'):
            s = (maxc - minc) / np.where(maxc == 0, 1.0, maxc)

        alpha_dark = (v < float(self.v_black_thresh)).astype(np.float32)
        alpha_bright_gray = ((v > float(self.v_white_thresh)) & (s < float(self.s_low_thresh))).astype(np.float32)

        # combine
        alpha_bg = np.maximum(alpha_bg1, np.maximum(alpha_dark, alpha_bright_gray))

        # composite over white
        out = (1.0 * alpha_bg[..., None] + norm * (1.0 - alpha_bg[..., None]))
        out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")

    # >>> NEW:
    def _otsu_threshold(self, gray_np):
        """Compute Otsu threshold (0-255) for a uint8 grayscale np array."""
        hist = np.bincount(gray_np.ravel(), minlength=256).astype(np.float64)
        total = gray_np.size
        cum_weights = np.cumsum(hist)
        cum_means = np.cumsum(hist * np.arange(256))
        global_mean = cum_means[-1] / total

        # Between-class variance
        denom = cum_weights * (total - cum_weights)
        denom[denom == 0] = 1.0
        var_between = (global_mean * cum_weights - cum_means) ** 2 / denom
        return int(np.argmax(var_between))

    # >>> NEW:
    def to_binary(self, image):
        """
        Convert PIL image to binary with white background (255) and black lines (0).
        Uses Otsu by default, or a fixed threshold.
        """
        gray = image.convert("L")
        arr = np.array(gray)

        if self.binarize_method.lower() == "otsu":
            t = self._otsu_threshold(arr)
        else:
            t = int(self.binarize_threshold)

        bin_arr = (arr > t).astype(np.uint8) * 255  # dark lines → 0, light bg → 255

        bin_img_L = Image.fromarray(bin_arr, mode="L")
        if self.channel_mode.upper() == "L":
            return bin_img_L
        else:
            # replicate to 3 channels if caller wants "RGB"
            return Image.merge("RGB", (bin_img_L, bin_img_L, bin_img_L))

    def __getitem__(self, idx):
        input_filename, gt_filename = self.pairs[idx]
        input_path = os.path.join(self.input_dir, input_filename)
        input_image = Image.open(input_path).convert(self.channel_mode)

        gt_image = None
        if gt_filename:
            gt_path = os.path.join(self.ground_truth_dir, gt_filename)
            gt_image = Image.open(gt_path).convert(self.channel_mode)

        if self.remove_background:
            #input_image = self.remove_bg(input_image)
            if gt_image:
                gt_image = self.remove_bg(gt_image)
        # >>> NEW: binarize before transforms
        if self.binarize:
            input_image = self.to_binary(input_image)
            if gt_image:
                gt_image = self.to_binary(gt_image)

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
            orig_w, orig_h = input_image.size

            if self.transform:
                input_image = self.transform(input_image)

            return input_image, input_filename, (orig_w, orig_h)  # Returning only image and filename
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
