#!/usr/bin/env python3
"""Process a dataset with the trained neural network (refactored, no docopt).

This version:
- Removes `docopt` in favor of `argparse` (or use as a library).
- Wraps the core pipeline into a `WireframeProcessor` class.
- Separates plotting into a dedicated `ResultPlotter` class.
- Keeps behavior close to the original script.

Example (CLI):
    python process_refactored.py config/wireframe.yaml checkpoints/ckpt.pth --devices 0 --plot
"""

import os
import os.path as osp
import argparse
import pprint
import random
from typing import Optional, Dict, Any

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio

from src.dhlp import lcnn
from src.dhlp.lcnn.utils import recursive_to
from src.dhlp.lcnn.config import C, M
from src.dhlp.lcnn.datasets import WireframeDataset, collate
from src.dhlp.lcnn.models.line_vectorizer import LineVectorizer
from src.dhlp.lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
from src.dhlp.lcnn.models.HT import hough_transform


class ResultPlotter:
    """Encapsulates plotting logic so it can be reused or swapped out."""
    def __init__(self, vmin: float = 0.4, vmax: float = 1.0, cmap_name: str = "jet"):
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        self.sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
        self.sm.set_array([])

    def color(self, x: float):
        return self.sm.to_rgba(x)

    def plot_sample(self, image_tensor: torch.Tensor, preds: Dict[str, torch.Tensor]) -> None:
        """Plot one sample.

        Args:
            image_tensor: CxHxW tensor (unnormalized as in the dataset).
            preds: dict containing at least 'lines' [N, 2, 2] and 'score' [N].
        """
        im = image_tensor.cpu().numpy().transpose(1, 2, 0)
        im = im * M.image.stddev + M.image.mean
        plt.imshow(np.clip(im, 0, 1))

        lines = preds["lines"].cpu().numpy() * 4
        scores = preds["score"].cpu().numpy()

        if len(lines) > 0 and not (lines[0] == 0).all():
            for i, ((a, b), s) in enumerate(zip(lines, scores)):
                if i > 0 and (lines[i] == lines[0]).all():
                    break
                plt.plot([a[1], b[1]], [a[0], b[0]], c=self.color(float(s)), linewidth=2)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


class WireframeProcessor:
    def __init__(
        self,
        config_file: str,
        checkpoint_path: str,
        devices: str = "0",
        seed: int = 0,
        plotter: Optional[ResultPlotter] = None,
    ):
        self.config_file = config_file
        self.checkpoint_path = checkpoint_path
        self.devices = devices
        self.seed = seed
        self.plotter = plotter

        # Load config
        C.update(C.from_yaml(filename=self.config_file))
        M.update(C.model)
        pprint.pprint(C, indent=4)

        # Seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.devices)
        if torch.cuda.is_available():
            device_name = "cuda"
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed(self.seed)
            print("Let's use", torch.cuda.device_count(), "GPU(s)!")
        else:
            device_name = "cpu"
            print("CUDA is not available")
        self.device = torch.device(device_name)

        # Vote index for Hough transform
        self.vote_index = self._load_or_make_vote_index()

        # Build & load model
        self.model = self._build_model()
        self._load_checkpoint()

        # Data
        self.loader = self._make_dataloader()

        # Output
        self.output_dir = C.io.outdir
        os.makedirs(self.output_dir, exist_ok=True)

    # ---------- Private helpers ----------

    def _load_or_make_vote_index(self) -> torch.Tensor:
        """Load or create the vote_index tensor used by the Hough transform."""
        if osp.isfile(C.io.vote_index):
            vote_index = sio.loadmat(C.io.vote_index)["vote_index"]
        else:
            # default settings: (128, 128, 3, 1)
            vote_index = hough_transform(rows=128, cols=128, theta_res=3, rho_res=1)
            sio.savemat(C.io.vote_index, {"vote_index": vote_index})
        vote_index = torch.from_numpy(vote_index).float().contiguous().to(self.device)
        print("load vote_index", tuple(vote_index.shape))
        return vote_index

    def _build_model(self) -> torch.nn.Module:
        if M.backbone == "stacked_hourglass":
            base = lcnn.models.hg(
                depth=M.depth,
                head=MultitaskHead,
                num_stacks=M.num_stacks,
                num_blocks=M.num_blocks,
                num_classes=sum(sum(M.head_size, [])),
                vote_index=self.vote_index,
            )
        else:
            raise NotImplementedError(f"Unsupported backbone: {M.backbone}")

        model = MultitaskLearner(base)
        model = LineVectorizer(model)
        model = model.to(self.device)
        model.eval()
        return model

    def _load_checkpoint(self) -> None:
        print(f"Loading checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def _make_dataloader(self) -> torch.utils.data.DataLoader:
        dataset = WireframeDataset(rootdir=C.io.datadir, split="test")
        loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=M.batch_size,
            collate_fn=collate,
            num_workers=C.io.num_workers if os.name != "nt" else 0,
            pin_memory=True,
        )
        return loader

    # ---------- Public API ----------

    @torch.inference_mode()
    def run(self, do_plot: bool = False) -> None:
        """Run inference over the test split, save .npz results, and optionally plot."""
        for batch_idx, (image, meta, target) in enumerate(self.loader):
            input_dict = {
                "image": recursive_to(image, self.device),
                "meta": recursive_to(meta, self.device),
                "target": recursive_to(target, self.device),
                "mode": "validation",
            }
            preds = self.model(input_dict)["preds"]
            for i in range(len(image)):
                # File name consistent with dataset filelist
                img_idx = batch_idx * M.batch_size + i
                image_filename = osp.splitext(osp.basename(self.loader.dataset.filelist[img_idx]))[0]

                # Save .npz
                npz_path = osp.join(self.output_dir, f"{image_filename}.npz")
                np.savez(npz_path, **{k: v[i].cpu().numpy() for k, v in preds.items()})
                print(f"Saved: {npz_path}")

                # Optional plot
                if do_plot and self.plotter is not None:
                    self.plotter.plot_sample(image[i], {k: v[i] for k, v in preds.items()})


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Process dataset with trained net (no docopt)")
    p.add_argument("yaml_config", help="Path to the YAML hyper-parameter file")
    p.add_argument("checkpoint", help="Path to the model checkpoint")
    p.add_argument("--devices", "-d", default="0", help="Comma separated GPU devices (default: 0)")
    p.add_argument("--plot", action="store_true", help="Plot results per image")
    return p


def main(argv: Optional[list] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    processor = WireframeProcessor(
        config_file=args.yaml_config,
        checkpoint_path=args.checkpoint,
        devices=args.devices,
        seed=0,
        plotter=ResultPlotter() if args.plot else None,
    )
    processor.run(do_plot=args.plot)


if __name__ == "__main__":
    main()
