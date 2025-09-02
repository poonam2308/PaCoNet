# line_prediction.py
import torch
import numpy as np
import os
import os.path as osp
import skimage.io
import skimage.transform
import json
import matplotlib.pyplot as plt
from src.dhlp import lcnn
from src.dhlp.dataset.gen_mask import generate_binary_mask
from src.dhlp.lcnn.models.line_vectorizer import LineVectorizer
from src.dhlp.lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
from src.dhlp.lcnn.models.HT import hough_transform
from src.dhlp.lcnn.postprocess import postprocess
from src.dhlp.lcnn.config import C, M


import cv2  # NEW

def line_passes_mask(a, b, mask, min_frac=0.6, samples=50):
    """
    a, b: endpoints in [y, x] (float, original image coords)
    mask: HxW uint8 (1 where valid)
    A line is 'inside' if at least min_frac of sampled points fall on mask==1.
    """
    H, W = mask.shape
    ys = np.linspace(a[0], b[0], samples)
    xs = np.linspace(a[1], b[1], samples)
    inside = 0
    valid = 0
    for yy, xx in zip(ys, xs):
        y = int(round(yy))
        x = int(round(xx))
        if 0 <= y < H and 0 <= x < W:
            valid += 1
            if mask[y, x] > 0:
                inside += 1
    if valid == 0:
        return False
    return (inside / valid) >= min_frac

class ThresholdPredictor(torch.nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


def save_line_predictions(im, lines, scores, base_name, tag, output_dir,
                          thr=0.5, mask=None, min_frac=0.6):
    """
    Save predictions (JSON + SVG).
    - im: numpy HxWx3 (original image, 0-255)
    - lines: Nx2x2 in [y,x]
    - scores: N scores aligned with lines
    - tag: string identifier ("pre", "mask", "post", etc.)
    - mask: optional binary mask
    """
    filtered_lines = []
    for (a, b), s in zip(lines, scores):
        if s < thr:
            continue
        if mask is not None and not line_passes_mask(a, b, mask, min_frac=min_frac):
            continue
        filtered_lines.append([[a[1], a[0]], [b[1], b[0]]])

    # --- JSON ---
    json_data = {"filename": base_name, "lines": filtered_lines}
    json_path = osp.join(output_dir, f"{osp.splitext(base_name)[0]}_{tag}.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)

    # --- SVG ---
    fig, ax = plt.subplots()
    ax.set_facecolor("white")
    ax.set_xlim(0, im.shape[1])
    ax.set_ylim(im.shape[0], 0)
    ax.set_axis_off()

    def get_color(x, y, image):
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            c = image[y, x]
            if c.max() > 1:
                c = c / 255.0
            return c[:3]
        return np.array([0.0, 0.0, 0.0])

    for (a, b), s in zip(lines, scores):
        if s < thr:
            continue
        if mask is not None and not line_passes_mask(a, b, mask, min_frac=min_frac):
            continue
        sx, sy = int(a[1]), int(a[0])
        mx, my = int((a[1] + b[1]) / 2), int((a[0] + b[0]) / 2)
        ex, ey = int(b[1]), int(b[0])
        col = (get_color(sx, sy, im) +
               get_color(mx, my, im) +
               get_color(ex, ey, im)) / 3.0
        ax.plot([a[1], b[1]], [a[0], b[0]], linewidth=2, color=tuple(col))

    svg_path = osp.join(output_dir, f"{osp.splitext(base_name)[0]}_{tag}.svg")
    plt.savefig(svg_path, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close()

    return svg_path, json_path


#
# def run_line_prediction_on_images_all(...):
#     # (network setup same as before)
#
#     results = {"pre": {"svgs": [], "jsons": []},
#                "mask": {"svgs": [], "jsons": []},
#                "post": {"svgs": [], "jsons": []},
#                "mask_post": {"svgs": [], "jsons": []}}
#
#     for im_path in image_paths:
#         im = skimage.io.imread(im_path)
#         if im.ndim == 2:
#             im = np.repeat(im[:, :, None], 3, 2)
#         im = im[:, :, :3]
#         im_resized = skimage.transform.resize(im, (512, 512)) * 255
#         image = (im_resized - M.image.mean) / M.image.stddev
#         image = torch.from_numpy(np.rollaxis(image, 2)[None].copy()).float().to(device)
#
#         with torch.no_grad():
#             H = net({...})["preds"]
#             lines = H["lines"][0].cpu().numpy() / 128 * im.shape[:2]
#             scores = H["score"][0].cpu().numpy()
#             effective_thr = float(score_threshold)
#
#         base_name = osp.basename(im_path)
#         mask = generate_binary_mask(im_path, white_thresh=250)
#
#         # Pre
#         svg, js = save_line_predictions(im, lines, scores, base_name, "pre",
#                                         output_dir, thr=effective_thr)
#         results["pre"]["svgs"].append(svg); results["pre"]["jsons"].append(js)
#
#         # Mask
#         svg, js = save_line_predictions(im, lines, scores, base_name, "mask",
#                                         output_dir, thr=effective_thr, mask=mask)
#         results["mask"]["svgs"].append(svg); results["mask"]["jsons"].append(js)
#
#         # Post
#         diag = (im.shape[0]**2 + im.shape[1]**2)**0.5
#         nlines, nscores = postprocess(lines, scores, diag*0.01, 0, False)
#         svg, js = save_line_predictions(im, nlines, nscores, base_name, "post",
#                                         output_dir, thr=effective_thr)
#         results["post"]["svgs"].append(svg); results["post"]["jsons"].append(js)
#
#         # Mask + Post
#         svg, js = save_line_predictions(im, nlines, nscores, base_name, "mask_post",
#                                         output_dir, thr=effective_thr, mask=mask)
#         results["mask_post"]["svgs"].append(svg); results["mask_post"]["jsons"].append(js)
#
#     return results

def run_line_prediction_on_images_all(
    image_paths,
    config_path="./src/dhlp/config/clust5kdenoisednew.yaml",
    checkpoint_path="./outputs/logs_clst5kdenew/250224-133604-baseline/checkpoint_best.pth",
    output_dir="./outputs/reals/redesigned",
    score_threshold=0.5,
):
    """
    Run line prediction with 4 result types:
    1. pre       = raw predictions (no filtering)
    2. mask      = raw + mask filtering
    3. post      = postprocess only
    4. mask_post = postprocess + mask filtering
    """
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

    # Initialize results
    results = {
        "pre": {"svgs": [], "jsons": []},
        "mask": {"svgs": [], "jsons": []},
        "post": {"svgs": [], "jsons": []},
        "mask_post": {"svgs": [], "jsons": []},
    }

    def save_outputs(im, raw_lines, raw_scores, base_name, tag, thr, mask=None, min_frac=0.6):
        # Filter lines
        filtered_lines = []
        for (a, b), s in zip(raw_lines, raw_scores):
            if s < thr:
                continue
            if mask is not None and not line_passes_mask(a, b, mask, min_frac=min_frac):
                continue
            filtered_lines.append([[a[1], a[0]], [b[1], b[0]]])

        # Save JSON
        json_data = {"filename": base_name, "lines": filtered_lines}
        json_path = osp.join(output_dir, f"{osp.splitext(base_name)[0]}_{tag}.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)

        # Save SVG
        fig, ax = plt.subplots()
        ax.set_facecolor("white")
        ax.set_xlim(0, im.shape[1])
        ax.set_ylim(im.shape[0], 0)
        ax.set_axis_off()

        def get_color(x, y, image):
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                c = image[y, x]
                if c.max() > 1:
                    c = c / 255.0
                return c[:3]
            return np.array([0.0, 0.0, 0.0])

        for (a, b), s in zip(raw_lines, raw_scores):
            if s < thr:
                continue
            if mask is not None and not line_passes_mask(a, b, mask, min_frac=min_frac):
                continue
            sx, sy = int(a[1]), int(a[0])
            mx, my = int((a[1] + b[1]) / 2), int((a[0] + b[0]) / 2)
            ex, ey = int(b[1]), int(b[0])
            col = (get_color(sx, sy, im) + get_color(mx, my, im) + get_color(ex, ey, im)) / 3.0
            ax.plot([a[1], b[1]], [a[0], b[0]], linewidth=2, color=tuple(col))

        svg_path = osp.join(output_dir, f"{osp.splitext(base_name)[0]}_{tag}.svg")
        plt.savefig(svg_path, bbox_inches="tight", facecolor="white", transparent=False)
        plt.close()

        return svg_path, json_path

    for im_path in image_paths:
        im = skimage.io.imread(im_path)
        if im.ndim == 2:
            im = np.repeat(im[:, :, None], 3, 2)
        im = im[:, :, :3]
        im_resized = skimage.transform.resize(im, (512, 512)) * 255
        image = (im_resized - M.image.mean) / M.image.stddev
        image = torch.from_numpy(np.rollaxis(image, 2)[None].copy()).float().to(device)

        with torch.no_grad():
            H = net({
                "image": image,
                "meta": [{"junc": torch.zeros(1, 2).to(device),
                          "jtyp": torch.zeros(1, dtype=torch.uint8).to(device),
                          "Lpos": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                          "Lneg": torch.zeros(2, 2, dtype=torch.uint8).to(device)}],
                "target": {"jmap": torch.zeros([1, 1, 128, 128]).to(device),
                           "joff": torch.zeros([1, 1, 2, 128, 128]).to(device)},
                "mode": "testing",
            })["preds"]

            lines = H["lines"][0].cpu().numpy() / 128 * im.shape[:2]
            scores = H["score"][0].cpu().numpy()
            effective_thr = float(score_threshold)

        base_name = osp.basename(im_path)
        mask = generate_binary_mask(im_path, white_thresh=250)

        # --- PRE ---
        svg, js = save_outputs(im, lines, scores, base_name, "pre", thr=effective_thr)
        results["pre"]["svgs"].append(svg); results["pre"]["jsons"].append(js)

        # --- MASK ---
        svg, js = save_outputs(im, lines, scores, base_name, "mask", thr=effective_thr, mask=mask)
        results["mask"]["svgs"].append(svg); results["mask"]["jsons"].append(js)

        # --- POST ---
        diag = (im.shape[0]**2 + im.shape[1]**2)**0.5
        nlines, nscores = postprocess(lines, scores, diag*0.01, 0, False)
        svg, js = save_outputs(im, nlines, nscores, base_name, "post", thr=effective_thr)
        results["post"]["svgs"].append(svg); results["post"]["jsons"].append(js)

        # --- MASK + POST ---
        svg, js = save_outputs(im, nlines, nscores, base_name, "mask_post", thr=effective_thr, mask=mask)
        results["mask_post"]["svgs"].append(svg); results["mask_post"]["jsons"].append(js)

    return results


def run_line_prediction_on_images_mask(image_paths, config_path="./src/dhlp/config/clust5kdenoisednew.yaml",
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

    # If you want to switch to learned threshold later, flip this flag to True.
    use_predicted_threshold = False
    threshold_predictor = ThresholdPredictor(input_dim=3).to(device)
    threshold_predictor.eval()

    # Keep track of all saved paths
    pre_svg_outputs, pre_json_outputs = [], []
    post_svg_outputs, post_json_outputs = [], []

    def save_outputs(im, raw_lines, raw_scores, base_name, tag, thr, mask=None, min_frac=0.6):
        """
        mask: optional HxW uint8 mask; when provided, lines must pass mask test
        min_frac: fraction of line samples that must lie on mask
        """
        # Filter lines by threshold (and mask if provided)
        filtered_lines = []
        for (a, b), s in zip(raw_lines, raw_scores):
            if s < thr:
                continue
            if mask is not None and not line_passes_mask(a, b, mask, min_frac=min_frac):
                continue
            # convert to [[x,y],[x,y]] for JSON
            filtered_lines.append([[a[1], a[0]], [b[1], b[0]]])

        # JSON
        json_data = {"filename": base_name, "lines": filtered_lines}
        json_path = osp.join(output_dir, f"{osp.splitext(base_name)[0]}_{tag}.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)

        # SVG — draw only the kept lines; color sampled from image
        fig, ax = plt.subplots()
        ax.set_facecolor("white")
        ax.set_xlim(0, im.shape[1])
        ax.set_ylim(im.shape[0], 0)
        ax.set_axis_off()

        def get_color(x, y, image):
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                c = image[y, x]
                if c.max() > 1:
                    c = c / 255.0
                return c[:3]
            return np.array([0.0, 0.0, 0.0])

        # re-draw from raw_lines/raw_scores but apply same filters for plotting
        for (a, b), s in zip(raw_lines, raw_scores):
            if s < thr:
                continue
            if mask is not None and not line_passes_mask(a, b, mask, min_frac=min_frac):
                continue
            sx, sy = int(a[1]), int(a[0])
            mx, my = int((a[1] + b[1]) / 2), int((a[0] + b[0]) / 2)
            ex, ey = int(b[1]), int(b[0])
            col = (get_color(sx, sy, im) + get_color(mx, my, im) + get_color(ex, ey, im)) / 3.0
            ax.plot([a[1], b[1]], [a[0], b[0]], linewidth=2, color=tuple(col))

        svg_path = osp.join(output_dir, f"{osp.splitext(base_name)[0]}_{tag}.svg")
        plt.savefig(svg_path, bbox_inches="tight", facecolor="white", transparent=False)
        plt.close()

        return svg_path, json_path

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

            # Scale to original image coordinates; note lines are in [y,x]
            lines = H["lines"][0].cpu().numpy() / 128 * im.shape[:2]
            scores = H["score"][0].cpu().numpy()

            # Optional learned threshold (off by default to match current behavior)
            image_feature = image.mean(dim=[2, 3])
            pred_thr = float(threshold_predictor(image_feature).cpu().item())
            effective_thr = pred_thr if use_predicted_threshold else float(score_threshold)

        base_name = osp.basename(im_path)

        # --- PRE: before postprocess ---
        pre_svg, pre_json = save_outputs(im, lines, scores, base_name, tag="pre", thr=effective_thr)
        pre_svg_outputs.append(pre_svg)
        pre_json_outputs.append(pre_json)

        # --- POST (now: MASKED): filter by image-derived binary mask, no geometric postprocess ---
        mask = generate_binary_mask(im_path, white_thresh=250)  # adjust threshold if needed
        # keep original raw lines/scores; filtering happens inside save_outputs via mask
        post_svg, post_json = save_outputs(
            im, lines, scores, base_name, tag="post", thr=effective_thr, mask=mask, min_frac=0.6
        )

        post_svg_outputs.append(post_svg)
        post_json_outputs.append(post_json)

    # Return both sets so callers can use them if needed
    return {
        "pre": {"svgs": pre_svg_outputs, "jsons": pre_json_outputs},
        "post": {"svgs": post_svg_outputs, "jsons": post_json_outputs},
    }



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

    # If you want to switch to learned threshold later, flip this flag to True.
    use_predicted_threshold = False
    threshold_predictor = ThresholdPredictor(input_dim=3).to(device)
    threshold_predictor.eval()

    # Keep track of all saved paths
    pre_svg_outputs, pre_json_outputs = [], []
    post_svg_outputs, post_json_outputs = [], []

    def save_outputs(im, raw_lines, raw_scores, base_name, tag, thr):
        """
        im: original HxWx3 image (numpy, 0-255)
        raw_lines: Nx2x2 in [y,x] order scaled to original image coords
        raw_scores: N scores aligned with raw_lines
        base_name: filename without extension
        tag: 'pre' or 'post'
        thr: threshold to filter scores
        """
        # Filter lines by threshold (same logic as before)
        filtered_lines = [
            [[a[1], a[0]], [b[1], b[0]]]   # convert to [[x,y],[x,y]]
            for (a, b), s in zip(raw_lines, raw_scores)
            if s >= thr
        ]

        # JSON
        json_data = {"filename": base_name, "lines": filtered_lines}
        json_path = osp.join(output_dir, f"{osp.splitext(base_name)[0]}_{tag}.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)

        # SVG — draw only the kept lines; color sampled from image
        fig, ax = plt.subplots()
        ax.set_facecolor("white")
        ax.set_xlim(0, im.shape[1])
        ax.set_ylim(im.shape[0], 0)
        ax.set_axis_off()

        def get_color(x, y, image):
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                c = image[y, x]
                if c.max() > 1:
                    c = c / 255.0
                return c[:3]
            return np.array([0.0, 0.0, 0.0])

        for (a, b), s in zip(raw_lines, raw_scores):
            if s < thr:
                continue
            # sample colors at start/mid/end, average to get line color
            sx, sy = int(a[1]), int(a[0])
            mx, my = int((a[1] + b[1]) / 2), int((a[0] + b[0]) / 2)
            ex, ey = int(b[1]), int(b[0])
            col = (get_color(sx, sy, im) + get_color(mx, my, im) + get_color(ex, ey, im)) / 3.0
            ax.plot([a[1], b[1]], [a[0], b[0]], linewidth=2, color=tuple(col))

        svg_path = osp.join(output_dir, f"{osp.splitext(base_name)[0]}_{tag}.svg")
        plt.savefig(svg_path, bbox_inches="tight", facecolor="white", transparent=False)
        plt.close()

        return svg_path, json_path

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

            # Scale to original image coordinates; note lines are in [y,x]
            lines = H["lines"][0].cpu().numpy() / 128 * im.shape[:2]
            scores = H["score"][0].cpu().numpy()

            # Optional learned threshold (off by default to match current behavior)
            image_feature = image.mean(dim=[2, 3])
            pred_thr = float(threshold_predictor(image_feature).cpu().item())
            effective_thr = pred_thr if use_predicted_threshold else float(score_threshold)

        base_name = osp.basename(im_path)

        # --- PRE: before postprocess ---
        pre_svg, pre_json = save_outputs(im, lines, scores, base_name, tag="pre", thr=effective_thr)
        pre_svg_outputs.append(pre_svg)
        pre_json_outputs.append(pre_json)

        # --- POST: apply postprocess as before, then save with same logic ---
        diag = (im.shape[0] ** 2 + im.shape[1] ** 2) ** 0.5
        nlines, nscores = postprocess(lines, scores, diag * 0.01, 0, False)

        post_svg, post_json = save_outputs(im, nlines, nscores, base_name, tag="post", thr=effective_thr)
        post_svg_outputs.append(post_svg)
        post_json_outputs.append(post_json)

    # Return both sets so callers can use them if needed
    return {
        "pre": {"svgs": pre_svg_outputs, "jsons": pre_json_outputs},
        "post": {"svgs": post_svg_outputs, "jsons": post_json_outputs},
    }


def run_line_prediction_on_images1(image_paths, config_path="./dhlp/config/clust5kdenoisednew.yaml",
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
