import argparse
import os
import os.path as osp
import pprint
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.transform
import torch
import yaml
import scipy.io as sio
import glob
import json
from matplotlib.widgets import Slider, Button
import lcnn
from lcnn.config import C, M
from lcnn.models.line_vectorizer import LineVectorizer
from lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
from lcnn.models.HT import hough_transform

from lcnn.postprocess import postprocess
from lcnn.utils import recursive_to

class ThresholdPredictor(torch.nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

PLTOPTS = {"color": "#33FFFF", "s": 15, "edgecolors": "none", "zorder": 5}
cmap = plt.get_cmap("plasma")
norm = mpl.colors.Normalize(vmin=0.9, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

score_percent = 0.01

def c(x):
    return sm.to_rgba(x)

def main():
    parser = argparse.ArgumentParser(description="Process an image with the trained neural network.")
    parser.add_argument("yaml_config", nargs="?",
                        default="config/clust5kdenoisednew.yaml",
                        help="Path to the yaml hyper-parameter file")
    parser.add_argument("checkpoint", nargs="?",
                        default="logs_and_results/logs_clst5kdenew/250224-133604-baseline/checkpoint_best.pth",
                        help="Path to the checkpoint")
    parser.add_argument("image_dir", nargs="?",
                        default="data/sd_data_redesign/input_images/1",
                        help="Directory containing input images")
    parser.add_argument("output_dir", nargs="?",
                        default="data/sd_data_redesign/output/1_",
                        help="Directory to save processed images")
    parser.add_argument("-d", "--devices", default="0", help="Comma-separated GPU devices [default: 0]")

    args = parser.parse_args()

    config_file = args.yaml_config
    source_dir = args.image_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    pprint.pprint(C, indent=4)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")
    device = torch.device(device_name)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    json_data = []
    # Load model
    if os.path.isfile(C.io.vote_index):
        vote_index = sio.loadmat(C.io.vote_index)['vote_index']
    else:
        vote_index = hough_transform(rows=128, cols=128, theta_res=3, rho_res=1)
        sio.savemat(C.io.vote_index, {'vote_index': vote_index})
    vote_index = torch.from_numpy(vote_index).float().contiguous().to(device)
    print('load vote_index', vote_index.shape)

    model = lcnn.models.hg(
        depth=M.depth,
        head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
        num_stacks=M.num_stacks,
        num_blocks=M.num_blocks,
        num_classes=sum(sum(M.head_size, [])),
        vote_index=vote_index,

    )
    model = MultitaskLearner(model)
    model = LineVectorizer(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # --- Load helper threshold predictor ---
    threshold_predictor = ThresholdPredictor(input_dim=3).to(device)
    threshold_predictor.eval()

    image_paths = glob.glob(osp.join(source_dir, "*.png"))
    # for imname in args["<images>"]:
    for imname in image_paths:
        print(f"Processing {imname}")
        im = skimage.io.imread(imname)
        if im.ndim == 2:
            im = np.repeat(im[:, :, None], 3, 2)
        im = im[:, :, :3]
        im_resized = skimage.transform.resize(im, (512, 512)) * 255
        image = (im_resized - M.image.mean) / M.image.stddev
        image = torch.from_numpy(np.rollaxis(image, 2)[None].copy()).float()
        with torch.no_grad():
            input_dict = {
                "image": image.to(device),
                "meta": [
                    {
                        "junc": torch.zeros(1, 2).to(device),
                        "jtyp": torch.zeros(1, dtype=torch.uint8).to(device),
                        "Lpos": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                        "Lneg": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                    }
                ],
                "target": {
                    "jmap": torch.zeros([1, 1, 128, 128]).to(device),
                    "joff": torch.zeros([1, 1, 2, 128, 128]).to(device),
                },
                "mode": "testing",
            }
            H = model(input_dict)["preds"]

        lines = H["lines"][0].cpu().numpy() / 128 * im.shape[:2]
        scores = H["score"][0].cpu().numpy()


        # json_data.append({
        #     "filename": osp.basename(imname),
        #     "lines": lines.tolist()
        # })
        # for i in range(1, len(lines)):
        #     if (lines[i] == lines[0]).all():
        #         lines = lines[:i]
        #         scores = scores[:i]
        #         break

        with torch.no_grad():
            image_feature = image.mean(dim=[2, 3]).to(device)  # [1, 3]
            predicted_threshold = threshold_predictor(image_feature).cpu().item()

        print(f"Predicted threshold: {predicted_threshold:.4f}")

        # postprocess lines to remove overlapped lines
        diag = (im.shape[0] ** 2 + im.shape[1] ** 2) ** 0.5
        nlines, nscores = postprocess(lines, scores, diag * 0.01, 0, False)
        # filtered_lines = [
        #     line.tolist() for line, score in zip(nlines, nscores) if score >= predicted_threshold
        # ]
        filtered_lines = [
            [[a[1], a[0]], [b[1], b[0]]]  # (x1, y1), (x2, y2)
            for (a, b), score in zip(nlines, nscores)
            if score >= predicted_threshold
        ]

        # json_data.append({
        #     "filename": osp.basename(imname),
        #     "lines": filtered_lines
        # })
        image_json={
            "filename": osp.basename(imname),
            "lines": filtered_lines
        }
        json_output_path = osp.join(output_dir, osp.basename(imname).replace(".png", ".json"))
        with open(json_output_path, "w") as json_file:
            json.dump(image_json, json_file, indent=4)

            # # --- Now start interactive plot ---
            # fig, ax = plt.subplots(figsize=(10, 10))
            # plt.subplots_adjust(bottom=0.25)
            # ax.set_facecolor("white")
            # ax.set_xlim(0, im.shape[1])
            # ax.set_ylim(im.shape[0], 0)
            # ax.set_axis_off()
            #
            # line_objs = []
            # for (a, b), s in zip(nlines, nscores):
            #     line_color = cmap(norm(s))
            #     l, = ax.plot([a[1], b[1]], [a[0], b[0]], color=line_color, linewidth=2)
            #     line_objs.append((l, s))
            #
            # # Add colorbar
            # plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            #
            # # --- Add slider ---
            # ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
            # slider = Slider(ax_slider, 'Threshold', 0.0, 1.0, valinit=0.01)
            #
            # # --- Add Save Button ---
            # saveax = plt.axes([0.8, 0.025, 0.1, 0.04])
            # button = Button(saveax, 'Save Image', hovercolor='0.975')
            #
            # def update(val):
            #     threshold = slider.val
            #     for line, score in line_objs:
            #         line.set_visible(score >= threshold)
            #     fig.canvas.draw_idle()
            #
            # slider.on_changed(update)
            #
            # def save(event):
            #     save_path = osp.join(output_dir, osp.basename(imname).replace(".png", f"-thresh-{slider.val:.2f}.svg"))
            #     plt.savefig(save_path, bbox_inches="tight", facecolor="white", transparent=False)
            #     print(f"Saved: {save_path}")
            #
            # button.on_clicked(save)
            #
            # plt.show()

            fig, ax = plt.subplots()
            ax.set_facecolor("white")  # Change to "black" if you want a black background
            ax.set_xlim(0, im.shape[1])  # Set X limits to image width
            ax.set_ylim(im.shape[0], 0)  # Set Y limits to image height (invert Y-axis)
            ax.set_axis_off()  # Hide axes
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)

            for (a, b), s in zip(nlines, nscores):
                if s < predicted_threshold:
                    continue

                # Compute key points along the line (start, mid, end)
                start_x, start_y = int(a[1]), int(a[0])  # Start point
                mid_x, mid_y = int((a[1] + b[1]) / 2), int((a[0] + b[0]) / 2)  # Midpoint
                end_x, end_y = int(b[1]), int(b[0])  # End point


                def get_color(x, y, image):
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        color = image[y, x]
                        if color.max() > 1:  # Check if values are in 0-255 range
                            color = color / 255  # Normalize only if necessary
                        return color
                    return np.array([0, 0, 0])  # Default to black if out of bounds

                # Sample colors at multiple points along the line
                start_color = get_color(start_x, start_y, im)
                mid_color = get_color(mid_x, mid_y, im)
                end_color = get_color(end_x, end_y, im)

                # Compute the average color for the line
                line_color = (start_color + mid_color + end_color) / 3
                line_color_tuple = tuple(line_color[:3])  # Convert to (R, G, B) tuple

                # Plot the line using the extracted color
                ax.plot([a[1], b[1]], [a[0], b[0]], color=line_color_tuple, linewidth=2)
                ax.scatter(a[1], a[0], **PLTOPTS)
                ax.scatter(b[1], b[0], **PLTOPTS)

            output_path =osp.join(output_dir, osp.basename(imname).replace(".png", "-lines.svg"))
            # plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

            plt.savefig(output_path, bbox_inches="tight", facecolor="white", transparent=False)

            plt.close()


if __name__ == "__main__":
    main()
