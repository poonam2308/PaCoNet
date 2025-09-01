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
import json
from itertools import combinations
from multiprocessing import Pool
import cv2
import numpy as np
import skimage.draw
import time
from docopt import docopt
from scipy.ndimage import zoom

try:
    sys.path.append("")
    sys.path.append("..")
    from src.dhlp.lcnn.utils import parmap
except Exception:
    raise


def inrange(v, shape):
    return 0 <= v[0] < shape[0] and 0 <= v[1] < shape[1]


def to_int(x):
    return tuple(map(int, x))


def save_heatmap(prefix, image, lines):
    im_rescale = (512, 512)
    heatmap_scale = (128, 128)

    fy, fx = heatmap_scale[1] / image.shape[0], heatmap_scale[0] / image.shape[1]
    jmap = np.zeros((1,) + heatmap_scale, dtype=float)
    joff = np.zeros((1, 2) + heatmap_scale, dtype=float)
    lmap = np.zeros(heatmap_scale, dtype=float)

    lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)
    lines = lines[:, :, ::-1]

    junc = []
    jids = {}

    def jid(jun):
        jun = tuple(jun[:2])
        if jun in jids:
            return jids[jun]
        jids[jun] = len(junc)
        junc.append(np.array(jun + (0,)))
        return len(junc) - 1

    lnid = []
    lpos, lneg = [], []
    for v0, v1 in lines:
        lnid.append((jid(v0), jid(v1)))
        lpos.append([junc[jid(v0)], junc[jid(v1)]])

        vint0, vint1 = to_int(v0), to_int(v1)
        jmap[0][vint0] = 1
        jmap[0][vint1] = 1
        rr, cc, value = skimage.draw.line_aa(*to_int(v0), *to_int(v1))
        lmap[rr, cc] = np.maximum(lmap[rr, cc], value)

    for v in junc:
        vint = to_int(v[:2])
        joff[0, :, vint[0], vint[1]] = v[:2] - vint - 0.5

    llmap = zoom(lmap, [0.5, 0.5])
    lineset = set([frozenset(l) for l in lnid])
    for i0, i1 in combinations(range(len(junc)), 2):
        if frozenset([i0, i1]) not in lineset:
            v0, v1 = junc[i0], junc[i1]
            vint0, vint1 = to_int(v0[:2] / 2), to_int(v1[:2] / 2)
            rr, cc, value = skimage.draw.line_aa(*vint0, *vint1)
            lneg.append([v0, v1, i0, i1, np.average(np.minimum(value, llmap[rr, cc]))])

    if len(lneg) == 0:
        print("Warning: No negative samples found in save_heatmap. Skipping negative line processing.")
        return  # Avoid crashing
    lneg.sort(key=lambda l: -l[-1])

    junc = np.array(junc, dtype=float)
    Lpos = np.array(lnid, dtype=int)
    Lneg = np.array([l[2:4] for l in lneg][:4000], dtype=int)
    lpos = np.array(lpos, dtype=float)
    lneg = np.array([l[:2] for l in lneg[:2000]], dtype=float)

    image = cv2.resize(image, im_rescale)
    np.savez_compressed(
        f"{prefix}_label.npz",
        aspect_ratio=image.shape[1] / image.shape[0],
        jmap=jmap,  # [J, H, W]    Junction heat map
        joff=joff,  # [J, 2, H, W] Junction offset within each pixel
        lmap=lmap,  # [H, W]       Line heat map with anti-aliasing
        junc=junc,  # [Na, 3]      Junction coordinate
        Lpos=Lpos,  # [M, 2]       Positive lines represented with junction indices
        Lneg=Lneg,  # [M, 2]       Negative lines represented with junction indices
        lpos=lpos,  # [Np, 2, 3]   Positive lines represented with junction coordinates
        lneg=lneg,  # [Nn, 2, 3]   Negative lines represented with junction coordinates
    )
    cv2.imwrite(f"{prefix}.png", image)


def handle(data, data_root, data_output, batch ):
    image_path = os.path.join(data_root, "images", data["filename"])
    # Check if the image file exists before processing
    if not os.path.exists(image_path):
        print(f"Error: Image file '{data['filename']}' not found! Skipping.")
        return
    im = cv2.imread(image_path)
    if im is None:
        print(f"Warning: Unable to read image {data['filename']}. Skipping.")
        return

    prefix = data["filename"].split(".")[0]
    print(prefix)

    lines = np.array(data["lines"]).reshape(-1, 2, 2)
    os.makedirs(os.path.join(data_output, batch), exist_ok=True)

    lines0 = lines.copy()

    lines1 = lines.copy()
    lines1[:, :, 0] = im.shape[1] - lines1[:, :, 0]
    lines2 = lines.copy()
    lines2[:, :, 1] = im.shape[0] - lines2[:, :, 1]
    lines3 = lines.copy()
    lines3[:, :, 0] = im.shape[1] - lines3[:, :, 0]
    lines3[:, :, 1] = im.shape[0] - lines3[:, :, 1]

    path = os.path.join(data_output, batch, prefix)
    save_heatmap(f"{path}_0", im[::, ::], lines0)
    if batch != "valid":
        save_heatmap(f"{path}_1", im[::, ::-1], lines1)
        save_heatmap(f"{path}_2", im[::-1, ::], lines2)
        save_heatmap(f"{path}_3", im[::-1, ::-1], lines3)
    print("Finishing", os.path.join(data_output, batch, prefix))

def main():
    start_time = time.time()
    args = docopt(__doc__)
    data_root = args["<src>"]
    data_output = args["<dst>"]

    os.makedirs(data_output, exist_ok=True)
    for batch in ["train", "valid"]:
        anno_file = os.path.join(data_root, f"{batch}.json")

        with open(anno_file, "r") as f:
            dataset = json.load(f)

        # parmap(handle, dataset, 16)

        with Pool(processes=8) as pool:  # Adjust number of processes if needed
            pool.starmap(handle, [(data, data_root, data_output, batch) for data in dataset])
    end_time= time.time()
    total_time = end_time - start_time
    print(f" Processing completed in {total_time:.2f} seconds ({total_time / 60:.2f} minutes)\n")

if __name__ == "__main__":
    main()
