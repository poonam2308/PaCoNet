import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import sys
from torch.utils.data import DataLoader, random_split

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))  # 2 levels up from this file
sys.path.insert(0, project_root)

from src.pc.config import config
from src.pc.config.config import get_args, load_config
from src.pc.utils import ensure_directory_exists, save_loss_plot, unet_transformation
from src.pc.models.unet import UNetSD
from src.pc.data_gen.custom_dataset_unet import CustomHSVMatchingDataset, CustomDatasetUnetSD
from src.pc.run_epoch_unet import train_epoch_unet_womo, validate_epoch_unet_womo


class UNetTrainer:
    def __init__(self):
        parser = get_args()
        self.args = parser.parse_args()
        self.cfg = load_config(self.args.cfg)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._set_seed(self.args.seed)
        self._prepare_dirs()
        self._prepare_transforms()

        self.model = UNetSD(in_channels=3, out_channels=3).to(self.device)

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _prepare_dirs(self):
        ensure_directory_exists(config.save_dir)
        ensure_directory_exists(config.model_log_dir)
        ensure_directory_exists(config.plot_log_dir)

    def _prepare_transforms(self):
        self.transform = unet_transformation(self.args)

    def _load_dataset(self):

        dataset = CustomDatasetUnetSD(
            input_dir=self.cfg['paths']['m_color_sep_plots'],
            ground_truth_dir=self.cfg['paths']['m_gt_plots_cat_crops'],
            transform=self.transform,
            hsv_tolerance=0.15,
            remove_background=True

        )

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        dataloader_train = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        dataloader_val = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)
        return dataloader_train, dataloader_val

    def train(self, save_prefix="unet_sd_cluster_mse"):
        dataloader_train, dataloader_val = self._load_dataset()

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        train_losses, val_losses = [], []

        for epoch in range(self.args.num_epochs):
            train_loss = train_epoch_unet_womo(self.model, dataloader_train, criterion,
                                               optimizer, self.device)
            val_loss = validate_epoch_unet_womo(self.model, dataloader_val, criterion,
                                                epoch, config.save_dir, self.device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            if (epoch + 1) % 5 == 0:
                ckpt_path = os.path.join(config.model_log_dir, f"{save_prefix}_model_epoch{epoch + 1}.pth")
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"Model saved at epoch {epoch+1}: {ckpt_path}")
                plot_name = f"loss_{save_prefix}_epoch_{epoch+1}.png"
                save_loss_plot(train_losses, val_losses, config.plot_log_dir,
                               plot_name, len(train_losses), len(val_losses))

        final_ckpt = os.path.join(config.model_log_dir, f"final_{save_prefix}_model.pth")
        torch.save(self.model.state_dict(), final_ckpt)
        print(f"Final model saved: {final_ckpt}")
        save_loss_plot(train_losses, val_losses, config.plot_log_dir, f"final_{save_prefix}_loss.png",
                       self.args.num_epochs, self.args.num_epochs)
if __name__ == "__main__":
    trainer = UNetTrainer()
    trainer.train()
