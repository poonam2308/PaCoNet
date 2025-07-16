# line_prediction.py
import torch
import numpy as np
import os
import os.path as osp
import skimage.io
import skimage.transform
import json
import matplotlib.pyplot as plt
from dhlp import lcnn
from dhlp.lcnn.models.line_vectorizer import LineVectorizer
from dhlp.lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
from dhlp.lcnn.models.HT import hough_transform
from dhlp.lcnn.postprocess import postprocess
from dhlp.lcnn.utils import recursive_to
from dhlp.lcnn.config import C, M

class ThresholdPredictor(torch.nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def run_line_prediction_on_images(image_paths, config_path="./dhlp/config/clust5kdenoisednew.yaml",
                                  checkpoint_path="./outputs/logs_clst5kdenew/250224-133604-baseline/checkpoint_best.pth",
                                  output_dir="./outputs/reals/redesigned",
                                  score_threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C.update(C.from_yaml(filename=config_path))
    M.update(C.model)

    vote_index = hough_transform(rows=128, cols=128, theta_res=3, rho_res=1)
    vote_index = torch.from_numpy(vote_index).float().contiguous().to(device)

    net = lcnn.models.hg(
        depth=M.depth,
        head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
        num_stacks=M.num_stacks,
        num_blocks=M.num_blocks,
        num_classes=sum(sum(M.head_size, [])),
        vote_index=vote_index,
    )
    net = MultitaskLearner(net)
    net = LineVectorizer(net)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net = net.to(device)
    net.eval()

    threshold_predictor = ThresholdPredictor(input_dim=3).to(device)
    threshold_predictor.eval()

    svg_outputs = []
    json_outputs = []

    for im_path in image_paths:
        im = skimage.io.imread(im_path)
        if im.ndim == 2:
            im = np.repeat(im[:, :, None], 3, 2)
        im = im[:, :, :3]
        im_resized = skimage.transform.resize(im, (512, 512)) * 255
        image = (im_resized - M.image.mean) / M.image.stddev
        image = torch.from_numpy(np.rollaxis(image, 2)[None].copy()).float().to(device)

        with torch.no_grad():
            input_dict = {
                "image": image,
                "meta": [{"junc": torch.zeros(1, 2).to(device),
                          "jtyp": torch.zeros(1, dtype=torch.uint8).to(device),
                          "Lpos": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                          "Lneg": torch.zeros(2, 2, dtype=torch.uint8).to(device)}],
                "target": {"jmap": torch.zeros([1, 1, 128, 128]).to(device),
                           "joff": torch.zeros([1, 1, 2, 128, 128]).to(device)},
                "mode": "testing",
            }
            H = net(input_dict)["preds"]
            lines = H["lines"][0].cpu().numpy() / 128 * im.shape[:2]
            scores = H["score"][0].cpu().numpy()

            image_feature = image.mean(dim=[2, 3])
            predicted_threshold = threshold_predictor(image_feature).cpu().item()

        diag = (im.shape[0] ** 2 + im.shape[1] ** 2) ** 0.5
        nlines, nscores = postprocess(lines, scores, diag * 0.01, 0, False)

        filtered_lines = [
            [[a[1], a[0]], [b[1], b[0]]]
            for (a, b), score in zip(nlines, nscores)
            # if score >= predicted_threshold
            if score >= score_threshold

        ]

        # Save JSON
        json_data = {
            "filename": osp.basename(im_path),
            "lines": filtered_lines
        }
        json_path = osp.join(output_dir, osp.basename(im_path).replace(".png", ".json"))
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)
        json_outputs.append(json_path)

        # Save SVG
        fig, ax = plt.subplots()
        ax.set_facecolor("white")
        ax.set_xlim(0, im.shape[1])
        ax.set_ylim(im.shape[0], 0)
        ax.set_axis_off()

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

            ax.plot([a[1], b[1]], [a[0], b[0]], color=line_color_tuple, linewidth=2)
        svg_path = osp.join(output_dir, osp.basename(im_path).replace(".png", ".svg"))
        plt.savefig(svg_path, bbox_inches="tight", facecolor="white", transparent=False)
        plt.close()
        svg_outputs.append(svg_path)

    return svg_outputs, json_outputs
