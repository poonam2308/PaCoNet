#!/usr/bin/env python3
"""Process a dataset with the trained neural network
Usage:
    process.py [options] <yaml-config> <checkpoint>
    process.py (-h | --help )

Arguments:
   <yaml-config>                 Path to the yaml hyper-parameter file
   <checkpoint>                  Path to the checkpoint

Options:
   -h --help                     Show this screen.
   -d --devices <devices>        Comma seperated GPU devices [default: 0]
   --plot                        Plot the result
"""

import os
import pprint
import random
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))  # 2 levels up from this file
sys.path.insert(0, project_root)

import numpy as np
import torch
from docopt import docopt
import scipy.io as sio
import src.dhlp.lcnn
from src.dhlp.lcnn.utils import recursive_to
from src.dhlp.lcnn.config import C, M
from src.dhlp.lcnn.datasets import WireframeDataset, collate
from src.dhlp.lcnn.models.line_vectorizer import LineVectorizer
from src.dhlp.lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
from src.dhlp.lcnn.models.HT import hough_transform
from process_utils import compute_nearest_junction_offset_stats, compute_distribution_mae, \
    compute_distribution_mae_median, wasserstein_2d, chamfer_distance_2d, get_mask_for_index, \
    filter_lines_with_mask_heatmap, line_nms



# # masks path for the color + unet 1
MASK_ROOT = "data/pcw_test/masks"

# masks path for the cluster  + unet  2
# MASK_ROOT = "data/pcw_test_cls/masks"

# MASK_ROOT = "data/pcw_crops_test/masks"



# ---- soft toggles ----
USE_MASK = True   # set to False to ignore masks
USE_NMS  = True   # set to False to skip line_nms
# ----------------------

HEATMAP_H, HEATMAP_W = 128, 128

def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"] or "config/wireframe.yaml"
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    pprint.pprint(C, indent=4)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")
    device = torch.device(device_name)

    ### load vote_index matrix for Hough transform
    ### defualt settings: (128, 128, 3, 1)
    if os.path.isfile(C.io.vote_index):
        vote_index = sio.loadmat(C.io.vote_index)['vote_index']
    else:
        vote_index = hough_transform(rows=128, cols=128, theta_res=3, rho_res=1)
        sio.savemat(C.io.vote_index, {'vote_index': vote_index})
    vote_index = torch.from_numpy(vote_index).float().contiguous().to(device)
    print('load vote_index', vote_index.shape)

    if M.backbone == "stacked_hourglass":
        model = src.dhlp.lcnn.models.hg(
            depth=M.depth,
            head=MultitaskHead,
            num_stacks=M.num_stacks,
            num_blocks=M.num_blocks,
            num_classes=sum(sum(M.head_size, [])),
            vote_index=vote_index,
        )
    else:
        raise NotImplementedError

    checkpoint = torch.load(args["<checkpoint>"])
    model = MultitaskLearner(model)
    model = LineVectorizer(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    loader = torch.utils.data.DataLoader(
        # WireframeDataset(args["<image-dir>"], split="valid"),
        WireframeDataset(rootdir=C.io.datadir, split="test"),
        shuffle=False,
        batch_size=M.batch_size,
        collate_fn=collate,
        num_workers=C.io.num_workers if os.name != "nt" else 0,
        pin_memory=True,
    )

    output_dir = C.io.outdir
    os.makedirs(output_dir, exist_ok=True)

    # Output file to save the offsets
    output_file = "offset_results_cat_dist_c.txt"
    output_dir = "output_offsets"
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
    output_path = os.path.join(output_dir, output_file)

    all_start_offsets = []
    all_end_offsets = []

    # Open file for writing
    with open(output_path, "w") as f:
        f.write("Image_Index, Avg_Start_Offset, Avg_End_Offset\n")  # CSV-style header
        all_offset_errors = []  # Store errors for averaging
        all_offset_errors_median=[]

        for batch_idx, (image, meta, target) in enumerate(loader):
            with torch.no_grad():
                input_dict = {
                    "image": recursive_to(image, device),
                    "meta": recursive_to(meta, device),
                    "target": recursive_to(target, device),
                    "mode": "validation",
                }
                H = model(input_dict)["preds"]

                for i in range(len(image)):
                    index = batch_idx * M.batch_size + i
                    print(f'Processing Image Index: {index}')
                    # line_endpoints = H["lines"][i].cpu().numpy() * 1.75

                    # 1) raw predictions in heatmap coords
                    lines_i = H["lines"][i].detach().cpu()
                    scores_i = H["score"][i].detach().cpu()

                    if USE_MASK:
                        mask_i = get_mask_for_index(loader.dataset, index, MASK_ROOT)

                        lines_i, scores_i, _ = filter_lines_with_mask_heatmap(
                            lines_i, scores_i, mask_i,
                            min_frac_inside=0.5,
                            n_samples=16,
                        )

                    if USE_NMS:
                        lines_i, scores_i, _ = line_nms(
                            lines_i, scores_i,
                            dist_thresh=2.0,
                        )
                    line_endpoints = lines_i.numpy() * 1.75
                    if len(line_endpoints) == 0:
                        continue


                    pred_lines = [tuple(line.flatten()) for line in line_endpoints]
                    if pred_lines and len(meta[i]["junc"]) > 0:
                        gt_junctions = meta[i]["junc"].cpu().numpy() * 1.75
                        # Collect predicted endpoints
                        pred_points = []
                        for line in pred_lines:
                            pred_points.append(line[:2])
                            pred_points.append(line[2:])
                        pred_points = np.array(pred_points)

                        # GT junctions (already Nx2)
                        gt_points = gt_junctions
                        # print("pred min/max:", pred_points.min(axis=0), pred_points.max(axis=0))
                        # print("gt   min/max:", gt_points.min(axis=0), gt_points.max(axis=0))

                        if len(pred_points) > 0 and len(gt_points) > 0:
                            stats = compute_distribution_mae(pred_points, gt_points)
                            stats_med = compute_distribution_mae_median(pred_points, gt_points)
                            wd = wasserstein_2d(pred_points, gt_points)
                            # wd = sliced_wasserstein_2d(pred_points, gt_points, n_proj=128)
                            # print(f"SlicedWasserstein={wd:.3f}px")
                            cd = chamfer_distance_2d(pred_points, gt_points, squared=False)

                            print(
                                f"Image {index}: "
                                f"MAE(mean) = {stats['mae_mean']:.3f}, "
                                f"MAE(std) = {stats['mae_std']:.3f}"
                            )


                            print(
                                f"Image {index}: "
                                f"MedianErr = {stats_med['mae_center']:.3f}, "
                                f"MADerr = {stats_med['mae_spread_err']:.3f}, "
                                f"Wasserstein = {wd:.3f}",
                                 f"Chamfer = {cd:.3f}px"
                            )

                            stats_med["wasserstein"] = wd
                            stats_med["chamfer"] = cd
                            all_offset_errors_median.append(stats_med)

                            all_offset_errors.append(stats)
                            #all_offset_errors_median.append(stats_med)

        avg_mae_mean = np.mean([e["mae_mean"] for e in all_offset_errors])
        avg_mae_std = np.mean([e["mae_std"] for e in all_offset_errors])

        # avg_mae_mad = np.mean([e["mae_center"] for e in all_offset_errors])
        # avg_mae_mad = np.mean([e["mae_spread_err"] for e in all_offset_errors])

        # print("\nFinal Distribution-Level Metrics:")
        # print(f"Avg MAE of Median: {avg_mae_mad:.3f}")
        # print(f"Avg MAE of MAD : {avg_mae_mad:.3f}")

        # avg_median_err = np.mean([e["mae_center"] for e in all_offset_errors_median])
        # avg_mad_err = np.mean([e["mae_spread_err"] for e in all_offset_errors_median])
        avg_wd = np.mean([e["wasserstein"] for e in all_offset_errors_median])
        avg_chamfer    = np.mean([e["chamfer"] for e in all_offset_errors_median])

        print("\nFinal Distribution-Level Metrics:")
        print(f"Avg MAE of Mean: {avg_mae_mean:.3f}")
        print(f"Avg MAE of Std : {avg_mae_std:.3f}")
        # print(f"Avg Median Center Error : {avg_median_err:.3f}")
        # print(f"Avg MAD Spread Error   : {avg_mad_err:.3f}")
        print(f"Avg Wasserstein (2D)   : {avg_wd:.3f}")
        print(f"Avg Chamfer Distance    : {avg_chamfer:.3f}px")


if __name__ == "__main__":
    main()


