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
from process_utils import nearest_junction

def visualize_points(img_tensor,
                     gt_junc,
                     start_points,
                     end_points,
                     save_path,
                     title=None):
    """
    Visualize GT junctions and predicted line endpoints on the image.

    Coordinates are assumed to be (y, x), so we plot:
        x = coord[:, 1]
        y = coord[:, 0]

    img_tensor   : torch tensor (C, H, W)
    gt_junc      : (Ng, 3) or (Ng, 2) numpy array (y, x, ...)
    start_points : (Ns, 2) numpy array (y, x)
    end_points   : (Ne, 2) numpy array (y, x)
    save_path    : path for saving the PNG
    """
    # Convert image tensor to HxWxC
    img = img_tensor.cpu().numpy()
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))  # C,H,W -> H,W,C

    # Normalize image for display
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img_vis = (img - img_min) / (img_max - img_min)
    else:
        img_vis = img

    plt.figure(figsize=(6, 6))
    plt.imshow(img_vis)

    # GT junctions (black circles, open)
    if gt_junc is not None and gt_junc.size > 0:
        gt_pts = gt_junc[:, :2]
        plt.scatter(gt_pts[:, 1], gt_pts[:, 0],
                    s=10, marker='o', edgecolors='k', facecolors='none',
                    label='GT junction')

    # Predicted start points (green x)
    if start_points is not None and start_points.size > 0:
        plt.scatter(start_points[:, 1], start_points[:, 0],
                    s=10, marker='x', color='g', label='Pred start')

    # Predicted end points (blue +)
    if end_points is not None and end_points.size > 0:
        plt.scatter(end_points[:, 1], end_points[:, 0],
                    s=10, marker='+', color='b', label='Pred end')

    if title is not None:
        plt.title(title)

    plt.legend(loc='upper right', fontsize=8)
    plt.axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"] or "config/wireframe.yaml"
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    pprint.pprint(C, indent=4)

    num_plots = int(6)

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

    output_file = "offset_results_cls.txt"
    output_dir = "output_offsets"
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
    output_path = os.path.join(output_dir, output_file)

    vis_dir = "output_point_mae_vis_sing"
    os.makedirs(vis_dir, exist_ok=True)
    plots_done = 0  # <--- add this

    # Initialize lists to store offsets across all images
    all_start_offsets = []
    all_end_offsets = []

    # Open file for writing
    with open(output_path, "w") as f:
        f.write("Image_Index, Avg_Start_Offset, Avg_End_Offset\n")  # CSV-style header

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

                    # Extract predicted line endpoints
                    line_endpoints = H["lines"][i].cpu().numpy() * 4  # Scale back
                    # print("h " , H["lines"][0])
                    if len(line_endpoints) == 0:
                        continue  # Skip empty detections

                    # start_points = line_endpoints[:, 0, :]
                    # end_points = line_endpoints[:, 1, :]

                    # Ensure start point is always the leftmost (x=0) and end point is rightmost (x=511)
                    corrected_start_points = []
                    corrected_end_points = []

                    for line in line_endpoints:
                        pt1, pt2 = line  # Two points forming the line
                        if pt1[0] <= pt2[0]:  # Ensure the leftmost is the start
                            corrected_start_points.append(pt1)
                            corrected_end_points.append(pt2)
                        else:
                            corrected_start_points.append(pt2)  # Swap to maintain left-to-right order
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
                    ground_truth_junctions = meta[i]["junc"].cpu().numpy() * 4  # Scale back

                    # Compute offsets
                    start_offsets = np.array([
                        np.linalg.norm(start - nearest_junction(start, ground_truth_junctions))
                        for start in start_points
                    ])
                    end_offsets = np.array([
                        np.linalg.norm(end - nearest_junction(end, ground_truth_junctions))
                        for end in end_points
                    ])

                    MAX_DIST = 30.0  # pixels (choose based on image size)

                    start_offsets = np.clip(start_offsets, 0, MAX_DIST)
                    end_offsets = np.clip(end_offsets, 0, MAX_DIST)

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
                            image[i],  # tensor (C,H,W)
                            ground_truth_junctions,  # (Na,3)
                            start_points,  # (Ns,2)
                            end_points,  # (Ne,2)
                            vis_path,
                            title=title,
                        )
                        plots_done += 1

        # Compute overall average offsets
        overall_start_offset = np.mean(all_start_offsets) if len(all_start_offsets) > 0 else 0
        overall_end_offset = np.mean(all_end_offsets) if len(all_end_offsets) > 0 else 0

        # Write overall results to the file
        f.write(f"\nOverall_Avg_Start_Offset: {overall_start_offset:.3f}\n")
        f.write(f"Overall_Avg_End_Offset: {overall_end_offset:.3f}\n")

    # Print overall results
    print(f"\nFinal Results Saved to {output_path}")
    print(f"Overall Avg Start Offset: {overall_start_offset:.3f}")
    print(f"Overall Avg End Offset: {overall_end_offset:.3f}")

if __name__ == "__main__":
    main()
