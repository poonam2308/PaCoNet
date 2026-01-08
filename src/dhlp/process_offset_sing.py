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
import matplotlib.pyplot as plt

import src.dhlp.lcnn
from src.dhlp.lcnn.utils import recursive_to
from src.dhlp.lcnn.config import C, M
from src.dhlp.lcnn.datasets import WireframeDataset, collate
from src.dhlp.lcnn.models.line_vectorizer import LineVectorizer
from src.dhlp.lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
from src.dhlp.lcnn.models.HT import hough_transform
from process_utils import (nearest_junction, visualize_points, set_seed,
                           get_mask_for_index, filter_lines_with_mask_heatmap, line_nms, coverage_mae_junctions,
                           gt_lines_from_Lpos, match_count_lines, match_count_lines_nearest, count_gt_covered_by_pred)

# # masks path for the color + unet 1
# MASK_ROOT = "data/pcw_test/masks"

# masks path for the cluster  + unet  2
# MASK_ROOT = "data/pcw_test_cls/masks"

MASK_ROOT = "data/pcw_crops_test/masks"

# ---- soft toggles ----
USE_MASK = True
USE_NMS  = True
# ----------------------

HEATMAP_H, HEATMAP_W = 128, 128


def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"] or "config/wireframe.yaml"
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    pprint.pprint(C, indent=4)
    num_plots = int(6)

    set_seed(0)

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

    output_file = "offset_results_c.txt"
    output_dir = "output_offsets_mae"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    vis_dir = "output_point_mae_vis_sing_c_m"
    os.makedirs(vis_dir, exist_ok=True)
    plots_done = 0

    all_start_offsets = []
    all_end_offsets = []

    total_k = 0  # total GT lines across dataset
    total_k_minus_m = 0  # total missed GT lines across dataset

    with open(output_path, "w") as f:
        f.write("Image_Index, Avg_Start_Offset, Avg_End_Offset\n")

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
                    gt_lines = gt_lines_from_Lpos(meta[i], scale=1.75)  # (k,2,2) in (y,x)
                    pr_lines = line_endpoints.astype(np.float32)  # (l,2,2) in (y,x)

                    k_gt = gt_lines.shape[0]
                    m_match = count_gt_covered_by_pred(gt_lines, pr_lines, max_cost=5.0)  # choose threshold in pixels

                    total_k += k_gt
                    total_k_minus_m += (k_gt - m_match)

                    if len(line_endpoints) == 0:
                        continue
                    corrected_start_points = []
                    corrected_end_points = []

                    for line in line_endpoints:
                        pt1, pt2 = line
                        if pt1[0] <= pt2[0]:
                            corrected_start_points.append(pt1)
                            corrected_end_points.append(pt2)
                        else:
                            corrected_start_points.append(pt2)
                            corrected_end_points.append(pt1)

                    # for line in line_endpoints:
                    #     pt1, pt2 = line
                    #
                    #     # points are (y, x) so compare x -> index 1
                    #     if pt1[1] <= pt2[1]:
                    #         start, end = pt1, pt2
                    #     else:
                    #         start, end = pt2, pt1
                    #
                    #     corrected_start_points.append(start)
                    #     corrected_end_points.append(end)

                    start_points = np.array(corrected_start_points)
                    end_points = np.array(corrected_end_points)

                    # Get ground truth junctions
                    ground_truth_junctions = meta[i]["junc"].cpu().numpy() *1.75  # Scale back


                    # Compute offsets
                    start_offsets = np.array([
                        np.linalg.norm(start - nearest_junction(start, ground_truth_junctions))
                        for start in start_points
                    ])
                    end_offsets = np.array([
                        np.linalg.norm(end - nearest_junction(end, ground_truth_junctions))
                        for end in end_points
                    ])

                    # MAX_DIST = 30.0  # pixels (choose based on image size)
                    #
                    # start_offsets = np.clip(start_offsets, 0, MAX_DIST)
                    # end_offsets = np.clip(end_offsets, 0, MAX_DIST)

                    # Compute average offsets for this image
                    average_start_offset = np.mean(start_offsets) if len(start_offsets) > 0 else 0
                    average_end_offset = np.mean(end_offsets) if len(end_offsets) > 0 else 0

                    # Store for overall computation
                    all_start_offsets.extend(start_offsets)
                    all_end_offsets.extend(end_offsets)

                    # Write per-image results
                    f.write(f"{index}, {average_start_offset:.3f}, {average_end_offset:.3f}\n")
                    print(
                        f"Image {index}: Avg Start Offset = {average_start_offset:.3f}, Avg End Offset = {average_end_offset:.3f}")


                    if (num_plots == 0) or (plots_done < num_plots):
                        vis_path = os.path.join(vis_dir, f"img_{index:05d}.png")
                        title = (f"Idx {index} | "
                                 f"StartOff={average_start_offset:.2f}, "
                                 f"EndOff={average_end_offset:.2f}")
                        visualize_points(
                            image[i],
                            ground_truth_junctions,
                            start_points,
                            end_points,
                            vis_path,
                            title=title,
                        )
                        plots_done += 1

        # Compute overall average offsets
        overall_start_offset = np.mean(all_start_offsets) if len(all_start_offsets) > 0 else 0
        overall_end_offset = np.mean(all_end_offsets) if len(all_end_offsets) > 0 else 0

        global_mae = (total_k_minus_m / total_k) if total_k > 0 else 0.0
        print(f"Global MAE (sum(k-m)/sum(k)): {global_mae:.6f}")
        print(f"Total GT lines (sum k): {total_k} | Total missed (sum(k-m)): {total_k_minus_m}")

        # Write overall results to the file
        f.write(f"\nOverall_Avg_Start_Offset: {overall_start_offset:.3f}\n")
        f.write(f"Overall_Avg_End_Offset: {overall_end_offset:.3f}\n")

    # Print overall results
    print(f"\nFinal Results Saved to {output_path}")
    print(f"Overall Avg Start Offset: {overall_start_offset:.3f}")
    print(f"Overall Avg End Offset: {overall_end_offset:.3f}")

if __name__ == "__main__":
    main()
