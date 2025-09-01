#!/usr/bin/env python
"""Process Huang's wireframe dataset for L-CNN network
Usage:
    dataset/wireframe.py <src> <dst>
    dataset/wireframe.py (-h | --help )

Examples:
    python dataset/wireframe.py /datadir/wireframe data/wireframe

Arguments:
    <src>                Original data directory of Huang's wireframe dataset
    <dst>                Directory of the output

Options:
   -h --help             Show this screen.
"""

import os
import sys
import cv2
import numpy as np
import time
from docopt import docopt
import glob

try:
    sys.path.append("")
    sys.path.append("..")
    from src.dhlp.lcnn.utils import parmap
except Exception:
    raise

def generate_binary_mask(image_path, white_thresh=250):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (image_gray < white_thresh).astype(np.uint8)
    return mask

def process_and_save_masks(image_dir, save_dir, target_size=(128, 128)):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Create directory if it doesn't exist

    # Get all image paths and sort them
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))  # Change extension if needed

    for image_path in image_paths:
        try:
            mask = generate_binary_mask(image_path)
            # Save as png
            filename = os.path.splitext(os.path.basename(image_path))[0]
            save_png_path = os.path.join(save_dir, f"{filename}.png")
            cv2.imwrite(save_png_path, (mask * 255).astype(np.uint8))

            mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            #filename = os.path.splitext(os.path.basename(image_path))[0]
            # save_path = os.path.join(save_dir, f"{filename}_label.npz")
            # np.savez_compressed(save_path, mask=mask_resized)
            #
            #
            # print(f"Saved mask: {save_path}")
            print(f"Saved mask imge: {save_png_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def main():
    start_time = time.time()
    args = docopt(__doc__)
    data_root = args["<src>"]
    data_output = args["<dst>"]

    os.makedirs(data_output, exist_ok=True)
    process_and_save_masks(data_root, data_output)
    end_time= time.time()
    total_time = end_time - start_time
    print(f" Processing completed in {total_time:.2f} seconds ({total_time / 60:.2f} minutes)\n")

if __name__ == "__main__":
    main()
