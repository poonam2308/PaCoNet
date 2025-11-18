import torch
import numpy as np
from torch.utils.data import DataLoader
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))  # 2 levels up from this file
sys.path.insert(0, project_root)
from src.pc.config import config
from src.pc.config.config import get_args, load_config
from src.pc.utils import ensure_directory_exists, unet_transformation
from src.pc.data_gen.custom_dataset_unet import CustomTestDatasetSD
from src.pc.models.unet import UNetSD
from src.pc.run_epoch_unet import test_unetsd_cluster

def unet_collate_fn(batch):
    # drop failed samples
    batch = [b for b in batch if b is not None]
    # batch is a list of (img, filename, (W, H))
    images, filenames, sizes = zip(*batch)  # sizes: tuple of (W, H)

    images = torch.stack(images, dim=0)
    filenames = list(filenames)
    sizes = list(sizes)  # list of (W, H)

    return images, filenames, sizes


class UNetTester:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = unet_transformation()

    def run_color(self, input_dir, output_dir, unet_chkpt, description="Denoising", resize_to_original=False):
        dataset = CustomTestDatasetSD(input_dir=input_dir, transform=self.transform)
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=unet_collate_fn,
        )

        print(f"Testing on: {description or input_dir} | Samples: {len(dataset)}")

        model = UNetSD(in_channels=3, out_channels=3).to(self.device)
        model.load_state_dict(torch.load(unet_chkpt, map_location=self.device))
        model.eval()

        for epoch in range(1):
            test_unetsd_cluster(model, loader, self.device, output_dir, resize_to_original=resize_to_original)
        print(f"Finished testing: {description or input_dir}")

    def run_cluster(self, input_dir, output_dir, unet_chkpt, description="Denoising", resize_to_original=False):
        dataset = CustomTestDatasetSD(input_dir=input_dir, transform=self.transform)
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=unet_collate_fn,
        )

        print(f"Testing on: {description or input_dir} | Samples: {len(dataset)}")

        model = UNetSD(in_channels=3, out_channels=3).to(self.device)
        model.load_state_dict(torch.load(unet_chkpt, map_location=self.device))
        model.eval()

        for epoch in range(1):
            test_unetsd_cluster(model, loader, self.device, output_dir, resize_to_original=resize_to_original)
        print(f"Finished testing: {description or input_dir}")


