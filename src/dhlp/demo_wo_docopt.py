#!/usr/bin/env python3
import os
import os.path as osp
import pprint
import random
import sys
import glob
import json
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))  # 2 levels up
sys.path.insert(0, project_root)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.transform
import torch
import scipy.io as sio

from src.dhlp import lcnn
from src.dhlp.lcnn.config import C, M
from src.dhlp.lcnn.models.line_vectorizer import LineVectorizer
from src.dhlp.lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
from src.dhlp.lcnn.models.HT import hough_transform
from src.dhlp.lcnn.postprocess import postprocess

# --- Helper threshold predictor ---
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


class LineDemoRunner:
    """
    Wrapper around your original demo logic so you can do:

        runner = LineDemoRunner(config_file, checkpoint_file, devices="0")
        runner.run_on_image(image_path, output_dir)
        # or
        runner.run_on_dir(source_dir, output_dir)
    """

    def __init__(self, config_file, checkpoint_file, devices="0"):
        # ---- config, seeds, device ----
        C.update(C.from_yaml(filename=config_file))
        M.update(C.model)
        print("Loaded config:")
        pprint.pprint(C, indent=4)

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)
        if torch.cuda.is_available():
            device_name = "cuda"
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed(0)
            print("Let's use", torch.cuda.device_count(), "GPU(s)!")
        else:
            device_name = "cpu"
            print("CUDA is not available, using CPU.")
        self.device = torch.device(device_name)

        # ---- vote_index ----
        if os.path.isfile(C.io.vote_index):
            vote_index = sio.loadmat(C.io.vote_index)["vote_index"]
        else:
            vote_index = hough_transform(rows=128, cols=128, theta_res=3, rho_res=1)
            sio.savemat(C.io.vote_index, {"vote_index": vote_index})
        vote_index = torch.from_numpy(vote_index).float().contiguous().to(self.device)
        print("load vote_index", vote_index.shape)
        self.vote_index = vote_index

        # ---- model ----
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        model = lcnn.models.hg(
            depth=M.depth,
            head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
            num_stacks=M.num_stacks,
            num_blocks=M.num_blocks,
            num_classes=sum(sum(M.head_size, [])),
            vote_index=self.vote_index,
        )
        model = MultitaskLearner(model)
        model = LineVectorizer(model)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        self.model = model

        # ---- helper threshold predictor ----
        threshold_predictor = ThresholdPredictor(input_dim=3).to(self.device)
        threshold_predictor.eval()
        self.threshold_predictor = threshold_predictor

    def run_on_dir(self, source_dir, output_dir):
        """
        Process all PNG images in source_dir and write JSON/visualizations into output_dir.
        """
        os.makedirs(output_dir, exist_ok=True)
        image_paths = glob.glob(osp.join(source_dir, "*.png"))
        for imname in image_paths:
            print(f"Processing {imname}")
            self.run_on_image(imname, output_dir)

    def run_on_image(self, image_path, output_dir):
        """
        Process a single image and save:
          - <name>.json
          - <name>-lines.png
          - <name>-lines.svg

        Returns a dict with basic info (paths, threshold).
        """
        os.makedirs(output_dir, exist_ok=True)
        im = skimage.io.imread(image_path)
        if im.ndim == 2:
            im = np.repeat(im[:, :, None], 3, 2)
        im = im[:, :, :3]

        # --- preprocessing as in original code ---
        im_resized = skimage.transform.resize(im, (512, 512)) * 255
        image = (im_resized - M.image.mean) / M.image.stddev
        image = torch.from_numpy(np.rollaxis(image, 2)[None].copy()).float()

        with torch.no_grad():
            input_dict = {
                "image": image.to(self.device),
                "meta": [
                    {
                        "junc": torch.zeros(1, 2).to(self.device),
                        "jtyp": torch.zeros(1, dtype=torch.uint8).to(self.device),
                        "Lpos": torch.zeros(2, 2, dtype=torch.uint8).to(self.device),
                        "Lneg": torch.zeros(2, 2, dtype=torch.uint8).to(self.device),
                    }
                ],
                "target": {
                    "jmap": torch.zeros([1, 1, 128, 128]).to(self.device),
                    "joff": torch.zeros([1, 1, 2, 128, 128]).to(self.device),
                },
                "mode": "testing",
            }
            H = self.model(input_dict)["preds"]

        lines = H["lines"][0].cpu().numpy() / 128 * im.shape[:2]
        scores = H["score"][0].cpu().numpy()

        # ---- predict threshold ----
        with torch.no_grad():
            image_feature = image.mean(dim=[2, 3]).to(self.device)  # [1, 3]
            predicted_threshold = self.threshold_predictor(image_feature).cpu().item()

        print(f"Predicted threshold used: {predicted_threshold:.4f}")

        # ---- postprocess lines to remove overlapped lines ----
        diag = (im.shape[0] ** 2 + im.shape[1] ** 2) ** 0.5
        nlines, nscores = postprocess(lines, scores, diag * 0.01, 0, False)

        filtered_lines = [
            [[a[1], a[0]], [b[1], b[0]]]  # (x1, y1), (x2, y2)
            for (a, b), score in zip(nlines, nscores)
            if score >= predicted_threshold
        ]

        # ---- write JSON with lines ----
        image_json = {
            "filename": osp.basename(image_path),
            "lines": filtered_lines,
        }
        json_output_path = osp.join(
            output_dir, osp.basename(image_path).replace(".png", ".json")
        )
        with open(json_output_path, "w") as json_file:
            json.dump(image_json, json_file, indent=4)

        # ---- overlay lines on original image (PNG) ----
        H_img, W_img = im.shape[:2]
        dpi = 100
        fig = plt.figure(figsize=(W_img / dpi, H_img / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(im, origin="upper")
        ax.set_xlim(0, W_img)
        ax.set_ylim(H_img, 0)
        ax.set_axis_off()

        def get_color(x, y, image):
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                color = image[y, x]
                if color.max() > 1:  # handle uint8 images
                    color = color / 255.0
                return color
            return np.array([0, 0, 0])

        for (a, b), s in zip(nlines, nscores):
            if s < predicted_threshold:
                continue

            start_x, start_y = int(a[1]), int(a[0])
            mid_x, mid_y = int((a[1] + b[1]) / 2), int((a[0] + b[0]) / 2)
            end_x, end_y = int(b[1]), int(b[0])

            start_color = get_color(start_x, start_y, im)
            mid_color = get_color(mid_x, mid_y, im)
            end_color = get_color(end_x, end_y, im)
            line_color = tuple(((start_color + mid_color + end_color) / 3.0)[:3])

            ax.plot([a[1], b[1]], [a[0], b[0]], linewidth=2, color=line_color)
            ax.scatter([a[1], b[1]], [a[0], b[0]], **PLTOPTS)

        output_path_png = osp.join(
            output_dir, osp.basename(image_path).rsplit(".", 1)[0] + "-lines.png"
        )
        fig.savefig(output_path_png, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # ---- lines only on white background (SVG) ----
        fig, ax = plt.subplots()
        ax.set_facecolor("white")
        ax.set_xlim(0, im.shape[1])
        ax.set_ylim(im.shape[0], 0)
        ax.set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        for (a, b), s in zip(nlines, nscores):
            if s < predicted_threshold:
                continue

            start_x, start_y = int(a[1]), int(a[0])
            mid_x, mid_y = int((a[1] + b[1]) / 2), int((a[0] + b[0]) / 2)
            end_x, end_y = int(b[1]), int(b[0])

            start_color = get_color(start_x, start_y, im)
            mid_color = get_color(mid_x, mid_y, im)
            end_color = get_color(end_x, end_y, im)
            line_color = (start_color + mid_color + end_color) / 3
            line_color_tuple = tuple(line_color[:3])

            ax.plot([a[1], b[1]], [a[0], b[0]], color=line_color_tuple, linewidth=2)
            ax.scatter(a[1], a[0], **PLTOPTS)
            ax.scatter(b[1], b[0], **PLTOPTS)

        output_path_svg = osp.join(
            output_dir, osp.basename(image_path).replace(".png", "-lines.svg")
        )
        plt.savefig(output_path_svg, bbox_inches="tight", facecolor="white", transparent=False)
        plt.close()

        return {
            "image": image_path,
            "json": json_output_path,
            "png": output_path_png,
            "svg": output_path_svg,
            "threshold": predicted_threshold,
        }


# ---------------- CLI (optional) ----------------
def main():
    parser = argparse.ArgumentParser(description="Process images with LCNN line detector.")
    parser.add_argument("config", help="Path to yaml config file")
    parser.add_argument("checkpoint", help="Path to checkpoint .pth file")
    parser.add_argument(
        "--image",
        help="Single image to process (if provided, --source is ignored)",
        default=None,
    )
    parser.add_argument(
        "--source",
        help="Directory with .png images to process",
        default=None,
    )
    parser.add_argument(
        "--output",
        help="Directory to save outputs",
        default="./outputs/demo",
    )
    parser.add_argument(
        "--devices",
        help="Comma-separated GPU devices (like '0' or '0,1')",
        default="0",
    )

    args = parser.parse_args()

    runner = LineDemoRunner(
        config_file=args.config,
        checkpoint_file=args.checkpoint,
        devices=args.devices,
    )

    if args.image is not None:
        runner.run_on_image(args.image, args.output)
    elif args.source is not None:
        runner.run_on_dir(args.source, args.output)
    else:
        raise ValueError("You must provide either --image or --source.")


if __name__ == "__main__":
    main()
