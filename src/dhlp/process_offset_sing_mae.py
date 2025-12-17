#!/usr/bin/env python3
"""Process a dataset with the trained neural network and compute point-wise MAE.
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

# Adjust project root as in your original script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
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


def mae_points_nearest(gt_start_points, gt_end_points,
                       pred_start_points, pred_end_points):
    """
    Compute mean absolute error between GT and predicted endpoints.

    For each GT point (start and end), find the closest predicted point
    (start or end) in Euclidean distance, then compute MAE in (x, y)
    between the matched pairs.

    Inputs:
        gt_start_points : (Ng1, 2) or (0, 2) numpy array
        gt_end_points   : (Ng2, 2) or (0, 2) numpy array
        pred_start_points : (Np1, 2) or (0, 2) numpy array
        pred_end_points   : (Np2, 2) or (0, 2) numpy array

    Returns:
        scalar float (mean absolute error per coordinate).
    """
    # No GT points at all
    if (gt_start_points.size == 0) and (gt_end_points.size == 0):
        return 0.0

    # Stack GT points -> (G, 2)
    gt_pts_list = []
    if gt_start_points.size != 0:
        gt_pts_list.append(gt_start_points.astype(np.float32))
    if gt_end_points.size != 0:
        gt_pts_list.append(gt_end_points.astype(np.float32))
    gt_pts = np.vstack(gt_pts_list)  # (G, 2)

    # Stack predicted points -> (P, 2)
    pred_pts_list = []
    if pred_start_points.size != 0:
        pred_pts_list.append(pred_start_points.astype(np.float32))
    if pred_end_points.size != 0:
        pred_pts_list.append(pred_end_points.astype(np.float32))
    if len(pred_pts_list) == 0:
        # No predicted points
        return 0.0
    pred_pts = np.vstack(pred_pts_list)  # (P, 2)

    # Pairwise differences: (G, P, 2)
    diff = gt_pts[:, None, :] - pred_pts[None, :, :]
    # Euclidean distances: (G, P)
    dists = np.linalg.norm(diff, axis=2)

    # For each GT point, get index of nearest predicted point
    nearest_idx = np.argmin(dists, axis=1)  # (G,)
    matched_pred = pred_pts[nearest_idx]    # (G, 2)

    # Mean absolute error per coordinate (x, y)
    mae = np.mean(np.abs(matched_pred - gt_pts))
    return float(mae)


def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"] or "config/wireframe.yaml"
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    pprint.pprint(C, indent=4)

    # Fix random seeds
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Device setup
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

    # Load / build vote_index for Hough transform
    if os.path.isfile(C.io.vote_index):
        vote_index = sio.loadmat(C.io.vote_index)["vote_index"]
    else:
        vote_index = hough_transform(rows=128, cols=128, theta_res=3, rho_res=1)
        sio.savemat(C.io.vote_index, {"vote_index": vote_index})
    vote_index = torch.from_numpy(vote_index).float().contiguous().to(device)
    print("load vote_index", vote_index.shape)

    # Build model
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

    # Load checkpoint and wrap with MultitaskLearner + LineVectorizer
    checkpoint = torch.load(args["<checkpoint>"], map_location=device)
    model = MultitaskLearner(model)
    model = LineVectorizer(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Data loader (test split)
    loader = torch.utils.data.DataLoader(
        WireframeDataset(rootdir=C.io.datadir, split="test"),
        shuffle=False,
        batch_size=M.batch_size,
        collate_fn=collate,
        num_workers=C.io.num_workers if os.name != "nt" else 0,
        pin_memory=True,
    )

    # Output setup
    output_dir = "output_point_mae"
    os.makedirs(output_dir, exist_ok=True)
    output_file = "point_mae_results_c.txt"
    output_path = os.path.join(output_dir, output_file)

    all_image_mae = []

    with open(output_path, "w") as f:
        f.write("Image_Index, Point_MAE\n")

        for batch_idx, (image, meta, target) in enumerate(loader):
            with torch.no_grad():
                input_dict = {
                    "image": recursive_to(image, device),
                    "meta": recursive_to(meta, device),
                    "target": recursive_to(target, device),
                    "mode": "validation",
                }
                H = model(input_dict)["preds"]

                for i_img in range(len(image)):
                    index = batch_idx * M.batch_size + i_img
                    print(f"Processing Image Index: {index}")

                    # Predicted line endpoints (scaled back)
                    line_endpoints = H["lines"][i_img].cpu().numpy() * 4.0  # [N_lines, 2, 2]
                    if len(line_endpoints) == 0:
                        print(f"Image {index}: no predicted lines, Point_MAE = 0.000")
                        f.write(f"{index}, 0.000\n")
                        continue

                    # Ensure start is leftmost (smaller x) and end is rightmost
                    corrected_start_points = []
                    corrected_end_points = []
                    for line in line_endpoints:
                        pt1, pt2 = line  # two endpoints
                        if pt1[0] <= pt2[0]:
                            corrected_start_points.append(pt1)
                            corrected_end_points.append(pt2)
                        else:
                            corrected_start_points.append(pt2)
                            corrected_end_points.append(pt1)

                    pred_start_points = np.array(corrected_start_points, dtype=np.float32)  # (N_pred, 2)
                    pred_end_points = np.array(corrected_end_points, dtype=np.float32)      # (N_pred, 2)

                    # Ground-truth junctions [Na, 3] -> use (x, y)
                    gt_junc = meta[i_img]["junc"].cpu().numpy() * 4.0  # (Na, 3)
                    if gt_junc.size == 0:
                        print(f"Image {index}: no GT junctions, Point_MAE = 0.000")
                        f.write(f"{index}, 0.000\n")
                        continue

                    gt_points = gt_junc[:, :2].astype(np.float32)   # (Na, 2)

                    # Here we treat all junctions as one GT set (no start/end distinction)
                    gt_start_points = gt_points
                    gt_end_points = np.empty((0, 2), dtype=np.float32)

                    mae_img = mae_points_nearest(
                        gt_start_points, gt_end_points,
                        pred_start_points, pred_end_points,
                    )

                    all_image_mae.append(mae_img)
                    print(f"Image {index}: Point_MAE = {mae_img:.3f}")
                    f.write(f"{index}, {mae_img:.3f}\n")

        overall_mae = float(np.mean(all_image_mae)) if len(all_image_mae) > 0 else 0.0
        f.write(f"\nOverall_Point_MAE: {overall_mae:.3f}\n")

    print(f"\nFinal Results Saved to {output_path}")
    print(f"Overall Point MAE (GT junctions → nearest predicted endpoints): {overall_mae:.3f}")

#
# process_offset_sing.py → “average distance from each predicted
# endpoint to the nearest GT junction” (Euclidean distance, start and end reported separately).
#
# process_offset_sing_mae*.py → “mean absolute error of
# GT junction positions vs their nearest predicted endpoint”
# (MAE per coordinate, after matching each GT to its closest prediction).

if __name__ == "__main__":
    main()
