#!/usr/bin/env python3
"""Evaluate mAPJ for LCNN, AFM, and Wireframe
Usage:
    eval-mAPJ.py <path>...
    eval-mAPJ.py (-h | --help )

Examples:
    python eval-mAPJ.py logs/*

Arguments:
    <path>                           One or more directories that contain *.npz

Options:
   -h --help                         Show this screen.
"""


import glob
import os.path as osp
import numpy as np
from docopt import docopt

import src.dhlp.lcnn.models
from src.dhlp.lcnn.metric import mAPJ, post_jheatmap


# GT = "data/wireframe/valid/*.npz"
# IM = "data/wireframe/valid-images/*.jpg"
# WF = "/data/wirebase/result/junc/2/17"
# AFM = "/data/wirebase/result/wireframe/afm/*.npz"


# GT = "data/pcwireframe_clst5k/valid/*.npz"
# IM = "data/pcwireframe_clst5k/valid/*.png"


# GT = "data/pcwireframe_test/test/*.npz"
# IM = "data/pcwireframe_test/test/*.png"

## python eval-mAP.py results_ct5k1
# GT = "data/pcwireframe_ct5k1/test/*.npz"
# IM = "data/pcwireframe_ct5k1/test/*.png"

# python eval-mAP.py results_ct5kde1
# GT = "data/pcwireframe_ct5kde1/test/*.npz"
# IM = "data/pcwireframe_ct5kde1/test/*.png"

# # python eval-mAP.py results_clst5knew
# ( 0.3 of the filelist)
GT = "data/pcwireframe_clst5knew/test/*.npz"
IM = "data/pcwireframe_clst5knew/test/*.png"
# #
# #python eval-mAP.py results_clst5kdenew
# GT = "data/pcwireframe_clst5kdenew/test/*.npz"
# IM = "data/pcwireframe_clst5kdenew/test/*.png"


DIST = [0.5, 1.0, 2.0]

def evaluate_lcnn(im_list, gt_list, lcnn_list):
    all_junc = np.zeros((0, 3))
    all_offset_junc = np.zeros((0, 3))
    all_junc_ids = np.zeros(0, dtype=np.int32)
    all_jc_gt = []

    # Convert lists to dictionaries for filename-based alignment
    gt_dict = {osp.splitext(osp.basename(f))[0]: f for f in gt_list}
    lcnn_dict = {osp.splitext(osp.basename(f))[0]: f for f in lcnn_list}

    # Get only the filenames that exist in both gt and predictions
    common_files = sorted(set(gt_dict.keys()) & set(lcnn_dict.keys()))

    if not common_files:
        print("No matching files found between predictions and ground truth.")
        return 0  # Return 0 if no matches

    for i, filename in enumerate(common_files):
        gt_fn = gt_dict[filename]
        lcnn_fn = lcnn_dict[filename]

        with np.load(lcnn_fn) as npz:
            result = {name: arr for name, arr in npz.items()}
            jmap = result["jmap"]
            joff = result["joff"]

        with np.load(gt_fn) as npz:
            junc_gt = npz["junc"][:, :2]

        jun_c = post_jheatmap(jmap[0])
        all_junc = np.vstack((all_junc, jun_c))
        jun_o_c = post_jheatmap(jmap[0], offset=joff[0])
        all_offset_junc = np.vstack((all_offset_junc, jun_o_c))

        all_jc_gt.append(junc_gt)
        all_junc_ids = np.hstack((all_junc_ids, np.array([i] * len(jun_c))))

    all_junc_ids = all_junc_ids.astype(np.int64)
    ap_jc = mAPJ(all_junc, all_jc_gt, DIST, all_junc_ids)
    ap_joc = mAPJ(all_offset_junc, all_jc_gt, DIST, all_junc_ids)
    print(f"  {ap_jc:.1f} | {ap_joc:.1f}")

def evaluate_lcnn_meanoffsets(im_list, gt_list, lcnn_list):
    # define result array to aggregate (n x 3) where 3 is (x, y, score)
    all_junc = np.zeros((0, 3))
    all_offset_junc = np.zeros((0, 3))
    all_junc_ids = np.zeros(0, dtype=np.int32)
    all_jc_gt = []
    mean_offsets = []

    for i, (lcnn_fn, gt_fn) in enumerate(zip(lcnn_list, gt_list)):
        with np.load(lcnn_fn) as npz:
            result = {name: arr for name, arr in npz.items()}
            jmap = result["jmap"]
            joff = result["joff"]

        with np.load(gt_fn) as npz:
            junc_gt = npz["junc"][:, :2]

        jun_c = post_jheatmap(jmap[0])
        all_junc = np.vstack((all_junc, jun_c))
        jun_o_c = post_jheatmap(jmap[0], offset=joff[0])
        all_offset_junc = np.vstack((all_offset_junc, jun_o_c))

        all_jc_gt.append(junc_gt)
        all_junc_ids = np.hstack((all_junc_ids, np.array([i] * len(jun_c))))

        # Compute mean offset between original and offset junctions
        mean_offset = np.mean(np.linalg.norm(jun_o_c[:, :2] - jun_c[:, :2], axis=1))
        mean_offsets.append(mean_offset)

    all_junc_ids = all_junc_ids.astype(np.int64)
    ap_jc = mAPJ(all_junc, all_jc_gt, DIST, all_junc_ids)
    ap_joc = mAPJ(all_offset_junc, all_jc_gt, DIST, all_junc_ids)
    mean_offset_value = np.mean(mean_offsets)

    print(f"  {ap_jc:.1f} | {ap_joc:.1f} | Mean Offset: {mean_offset_value:.3f}")

def main():
    args = docopt(__doc__)
    gt_list = sorted(glob.glob(GT))
    im_list = sorted(glob.glob(IM))

    for path in args["<path>"]:
        print("Evaluating", path)
        lcnn_list = sorted(glob.glob(osp.join(path, "*.npz")))
        evaluate_lcnn(im_list, gt_list, lcnn_list)
        # evaluate_lcnn_meanoffsets(im_list, gt_list, lcnn_list)


if __name__ == "__main__":
    main()


#  Evaluating results_all
#   57.4 | 85.2

# Evaluating results_ct5k1
#   56.3 | 82.3

# Evaluating results_ct5kde1
#  56.4 | 81.9

# Evaluating results_clst5k


# Evaluating results_clst5kdenew
#   55.2 | 79.3



# Evaluating results_ct5kde1 with 80%
#   56.5 | 82.1




