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



class UNetTester:
    def __init__(self):
        parser = get_args()
        self.args = parser.parse_args()
        self.cfg = load_config(self.args.cfg)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_seed(self.args.seed)
        self._prepare_dirs()

        self.transform = unet_transformation(self.args)

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _prepare_dirs(self):
        ensure_directory_exists(config.model_log_dir)
        ensure_directory_exists(config.plot_log_dir)

    def run_color(self, input_dir, output_dir, description="Denoising"):
        dataset = CustomTestDatasetSD(input_dir=input_dir, transform=self.transform)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        print(f"Testing on: {description or input_dir} | Samples: {len(dataset)}")

        model = UNetSD(in_channels=3, out_channels=3).to(self.device)
        model.load_state_dict(torch.load(self.cfg['unet']['chkpt_path_color'], map_location=self.device))
        model.eval()

        for epoch in range(self.args.num_epochs):
            test_unetsd_cluster(model, loader, self.device, output_dir)
        print(f"Finished testing: {description or input_dir}")

    def run_cluster(self, input_dir, output_dir, description="Denoising"):
        dataset = CustomTestDatasetSD(input_dir=input_dir, transform=self.transform)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        print(f"Testing on: {description or input_dir} | Samples: {len(dataset)}")

        model = UNetSD(in_channels=3, out_channels=3).to(self.device)
        model.load_state_dict(torch.load(self.cfg['unet']['chkpt_path_cluster'], map_location=self.device))
        model.eval()

        for epoch in range(self.args.num_epochs):
            test_unetsd_cluster(model, loader, self.device, output_dir)
        print(f"Finished testing: {description or input_dir}")


if __name__ == "__main__":
    tester = UNetTester()
    input_dir = tester.cfg['paths']['m_color_sep_plots']
    output_dir = tester.cfg['unet']['output_dir_color']

    input_dir_cls = tester.cfg['paths']['m_cluster_sep_plots']
    output_dir_cls = tester.cfg['unet']['output_dir_cluster']

    # tester.run_color(input_dir, output_dir)
    tester.run_cluster(input_dir_cls, output_dir_cls)

