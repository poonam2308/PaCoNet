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
from process_utils import compute_nearest_junction_offset_stats


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
        WireframeDataset(rootdir=C.io.datadir, split="valid"),
        shuffle=False,
        batch_size=M.batch_size,
        collate_fn=collate,
        num_workers=C.io.num_workers if os.name != "nt" else 0,
        pin_memory=True,
    )

    output_dir = C.io.outdir
    os.makedirs(output_dir, exist_ok=True)

    # Output file to save the offsets
    output_file = "offset_stats25_sd.txt"
    output_dir = "output_offsets"
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
    output_path = os.path.join(output_dir, output_file)

    all_start_offsets = []
    all_end_offsets = []

    # Open file for writing
    with open(output_path, "w") as f:
        f.write("Image_Index, Avg_Start_Offset, Avg_End_Offset\n")  # CSV-style header
        all_offset_errors = []  # Store errors for averaging

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
                    line_endpoints = H["lines"][i].cpu().numpy() * 4

                    if len(line_endpoints) == 0:
                        continue

                    for idx in range(len(line_endpoints)):
                        if random.random() > 0.5:
                            line_endpoints[idx] = line_endpoints[idx][::-1]

                    pred_lines = [tuple(line.flatten()) for line in line_endpoints]
                    if pred_lines and len(meta[i]["junc"]) > 0:
                        gt_junctions = meta[i]["junc"].cpu().numpy() * 4
                        offset_stats = compute_nearest_junction_offset_stats(pred_lines, gt_junctions)
                        all_offset_errors.append(offset_stats)
                        # Log results
                        print(f"Image {index}: Mean Offset = {offset_stats['mean_offset']:.3f}, "
                              f"Std Dev = {offset_stats['std_offset']:.3f}, "
                              f"Lower Bound = {offset_stats['lower_bound']:.3f}, "
                              f"Upper Bound = {offset_stats['upper_bound']:.3f}")

                        f.write(f"{index}, {offset_stats['mean_offset']:.2f}, {offset_stats['std_offset']:.2f}, "
                                f"{offset_stats['lower_bound']:.2f}, {offset_stats['upper_bound']:.2f}\n")

        if all_offset_errors:
                # Compute overall averages based on nearest junction-based errors
                avg_mean_offset = np.mean([e["mean_offset"] for e in all_offset_errors])
                avg_lower_offset = np.mean([e["lower_bound"] for e in all_offset_errors])
                avg_upper_offset = np.mean([e["upper_bound"] for e in all_offset_errors])
                # avg_overall_offset = np.mean([e["overall_offset"] for e in all_offset_errors])

                # Write final averages to the output file
                f.write("\nOverall_Averages (Nearest Junction Based):\n")
                f.write(f"Avg Mean Offset: {round(avg_mean_offset, 2)}\n")
                f.write(f"Avg Lower Offset: {round(avg_lower_offset, 2)}\n")
                f.write(f"Avg Upper Offset: {round(avg_upper_offset, 2)}\n")
                # f.write(f"Avg Overall Offset: {round(avg_overall_offset, 2)}\n")

                # Print the final averages
                print("\nFinal Average Offset Errors (Nearest Junction Based):")
                print(f"Avg Mean Offset: {round(avg_mean_offset, 2)}")
                print(f"Avg Lower Offset: {round(avg_lower_offset, 2)}")
                print(f"Avg Upper Offset: {round(avg_upper_offset, 2)}")
            # print(f"Avg Overall Offset: {round(avg_overall_offset, 2)}")

    #
    #                 # Compute offsets
    #                 start_offsets = np.array([
    #                     np.linalg.norm(start - nearest_junction(start, ground_truth_junctions))
    #                     for start in start_points
    #                 ])
    #                 end_offsets = np.array([
    #                     np.linalg.norm(end - nearest_junction(end, ground_truth_junctions))
    #                     for end in end_points
    #                 ])
    #
    #                 # Compute average offsets for this image
    #                 average_start_offset = np.mean(start_offsets) if len(start_offsets) > 0 else 0
    #                 average_end_offset = np.mean(end_offsets) if len(end_offsets) > 0 else 0
    #
    #                 # Store for overall computation
    #                 all_start_offsets.extend(start_offsets)
    #                 all_end_offsets.extend(end_offsets)
    #
    #                 # Write per-image results
    #                 f.write(f"{index}, {average_start_offset:.3f}, {average_end_offset:.3f}\n")
    #                 print(
    #                     f"Image {index}: Avg Start Offset = {average_start_offset:.3f}, Avg End Offset = {average_end_offset:.3f}")
    #
    #     # Compute overall average offsets
    #     overall_start_offset = np.mean(all_start_offsets) if len(all_start_offsets) > 0 else 0
    #     overall_end_offset = np.mean(all_end_offsets) if len(all_end_offsets) > 0 else 0
    #
    #     # Write overall results to the file
    #     f.write(f"\nOverall_Avg_Start_Offset: {overall_start_offset:.3f}\n")
    #     f.write(f"Overall_Avg_End_Offset: {overall_end_offset:.3f}\n")
    #
    # # Print overall results
    # print(f"\nFinal Results Saved to {output_path}")
    # print(f"Overall Avg Start Offset: {overall_start_offset:.3f}")
    # print(f"Overall Avg End Offset: {overall_end_offset:.3f}")

#     for batch_idx, (image, meta, target) in enumerate(loader):
#         with torch.no_grad():
#             input_dict = {
#                 "image": recursive_to(image, device),
#                 "meta": recursive_to(meta, device),
#                 "target": recursive_to(target, device),
#                 "mode": "validation",
#             }
#             H = model(input_dict)["preds"]
#             for i in range(len(image)):
#                 index = batch_idx * M.batch_size + i
#                 print('index', index)
#                 np.savez(
#                     osp.join(output_dir, f"{index:06}.npz"),
#                     **{k: v[i].cpu().numpy() for k, v in H.items()},
#                 )
#                 if not args["--plot"]:
#                     continue
#                 im = image[i].cpu().numpy().transpose(1, 2, 0)
#                 im = im * M.image.stddev + M.image.mean
#                 lines = H["lines"][i].cpu().numpy() * 4
#                 scores = H["score"][i].cpu().numpy()
#                 if len(lines) > 0 and not (lines[0] == 0).all():
#                     for i, ((a, b), s) in enumerate(zip(lines, scores)):
#                         if i > 0 and (lines[i] == lines[0]).all():
#                             break
#                         plt.plot([a[1], b[1]], [a[0], b[0]], c=c(s), linewidth=4)
#                 plt.show()
#
#
# cmap = plt.get_cmap("jet")
# norm = mpl.colors.Normalize(vmin=0.4, vmax=1.0)
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
#
#
# def c(x):
#     return sm.to_rgba(x)
#
#
if __name__ == "__main__":
    main()
