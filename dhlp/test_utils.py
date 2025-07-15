import argparse
import json

from lcnn.config import C, M
import torch
import os
import scipy.io as sio
from lcnn.models.HT import hough_transform
from lcnn.models.line_vectorizer import LineVectorizer
from lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
import lcnn
import numpy as np
import skimage.io
import skimage.transform
import glob
import os.path as osp
from lcnn.postprocess import postprocess
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Process an image with the trained neural network.")
    parser.add_argument("yaml_config", nargs="?", default="config/clust5kdenoisednew.yaml")
    parser.add_argument("checkpoint", nargs="?", default="logs_and_results/.../checkpoint_best.pth")
    parser.add_argument("image_dir", nargs="?", default="data/real_plots/input_images/1")
    parser.add_argument("output_dir", nargs="?", default="data/predicted_data/output/1_")
    parser.add_argument("-d", "--devices", default="0")
    return parser.parse_args()

def setup_config(config_file):
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    return C, M


def get_vote_index(vote_index_path, device):
    if os.path.isfile(vote_index_path):
        vote_index = sio.loadmat(vote_index_path)['vote_index']
    else:
        vote_index = hough_transform(rows=128, cols=128, theta_res=3, rho_res=1)
        sio.savemat(vote_index_path, {'vote_index': vote_index})
    return torch.from_numpy(vote_index).float().to(device)

def load_model(checkpoint_path, vote_index, M, device):
    base_model = lcnn.models.hg(
        depth=M.depth,
        head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
        num_stacks=M.num_stacks,
        num_blocks=M.num_blocks,
        num_classes=sum(sum(M.head_size, [])),
        vote_index=vote_index,
    )
    model = LineVectorizer(MultitaskLearner(base_model))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def load_images(image_dir):
    return glob.glob(osp.join(image_dir, "*.png"))

def preprocess_image(image_path, M):
    im = skimage.io.imread(image_path)
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, axis=2)
    im = im[:, :, :3]
    im_resized = skimage.transform.resize(im, (512, 512)) * 255
    image = (im_resized - M.image.mean) / M.image.stddev
    tensor_image = torch.from_numpy(np.rollaxis(image, 2)[None].copy()).float()
    return im, tensor_image


def predict(model, image_tensor, device):
    input_dict = {
        "image": image_tensor.to(device),
        "meta": [{"junc": torch.zeros(1, 2).to(device), "jtyp": torch.zeros(1, dtype=torch.uint8).to(device),
                  "Lpos": torch.zeros(2, 2, dtype=torch.uint8).to(device), "Lneg": torch.zeros(2, 2, dtype=torch.uint8).to(device)}],
        "target": {
            "jmap": torch.zeros([1, 1, 128, 128]).to(device),
            "joff": torch.zeros([1, 1, 2, 128, 128]).to(device),
        },
        "mode": "testing",
    }
    with torch.no_grad():
        return model(input_dict)["preds"]


def postprocess_lines(H, im_shape, threshold_predictor, image_tensor, device):
    lines = H["lines"][0].cpu().numpy() / 128 * im_shape[:2]
    scores = H["score"][0].cpu().numpy()

    with torch.no_grad():
        image_feature = image_tensor.mean(dim=[2, 3]).to(device)
        predicted_threshold = threshold_predictor(image_feature).cpu().item()

    diag = (im_shape[0] ** 2 + im_shape[1] ** 2) ** 0.5
    nlines, nscores = postprocess(lines, scores, diag * 0.01, 0, False)
    filtered_lines = [
        [[a[1], a[0]], [b[1], b[0]]]
        for (a, b), score in zip(nlines, nscores)
        if score >= predicted_threshold
    ]
    return filtered_lines, predicted_threshold, nlines, nscores


def plot_lines(im, nlines, nscores, threshold, output_path):
    fig, ax = plt.subplots()
    ax.set_facecolor("white")
    ax.set_xlim(0, im.shape[1])
    ax.set_ylim(im.shape[0], 0)
    ax.set_axis_off()

    def get_color(x, y):
        if 0 <= x < im.shape[1] and 0 <= y < im.shape[0]:
            color = im[y, x]
            return color / 255 if color.max() > 1 else color
        return np.array([0, 0, 0])

    for (a, b), s in zip(nlines, nscores):
        if s < threshold:
            continue
        color = (get_color(*map(int, a[::-1])) +
                 get_color(*map(int, ((a + b) / 2)[::-1])) +
                 get_color(*map(int, b[::-1]))) / 3
        ax.plot([a[1], b[1]], [a[0], b[0]], color=color, linewidth=2)

    plt.savefig(output_path, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close()


def main():
    args = parse_args()
    C, M = setup_config(args.yaml_config)

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vote_index = get_vote_index(C.io.vote_index, device)
    model = load_model(args.checkpoint, vote_index, M, device)
    threshold_predictor = ThresholdPredictor(input_dim=3).to(device).eval()

    for img_path in load_images(args.image_dir):
        im, image_tensor = preprocess_image(img_path, M)
        H = predict(model, image_tensor, device)
        filtered_lines, thresh, nlines, nscores = postprocess_lines(H, im.shape, threshold_predictor, image_tensor, device)

        json_output_path = osp.join(args.output_dir, osp.basename(img_path).replace(".png", ".json"))
        with open(json_output_path, "w") as f:
            json.dump({"filename": osp.basename(img_path), "lines": filtered_lines}, f, indent=4)

        output_svg_path = osp.join(args.output_dir, osp.basename(img_path).replace(".png", "-lines.svg"))
        plot_lines(im, nlines, nscores, thresh, output_svg_path)

if __name__ == "__main__":
    main()
