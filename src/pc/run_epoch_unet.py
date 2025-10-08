import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import torchvision.transforms as transforms


def _denorm01(arr):
    # arr float in CHW/HWC; if in [-1,1] -> [0,1]; if already [0,1], leave it
    if arr.min() < 0.0:     # robust test for normalized tensors
        arr = arr * 0.5 + 0.5
    return np.clip(arr, 0.0, 1.0)

def _composite_over_white(arr01):
    # arr01 HxWx(3|4) float in [0,1]
    if arr01.shape[-1] == 4:
        rgb = arr01[..., :3]
        a = arr01[..., 3:4]
        return rgb * a + (1.0 - a) * 1.0
    return arr01[..., :3]

def _to_uint8_rgb(arr):
    # arr HxWxC float in [0,1] or uint8; returns HxWx3 uint8, alpha composited
    if arr.dtype != np.uint8:
        arr = np.asarray(arr, dtype=np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
    if arr.shape[-1] == 4:  # compose if RGBA
        a = (arr[..., 3:4].astype(np.float32) / 255.0)
        rgb = arr[..., :3].astype(np.float32)
        arr = (rgb * a + (1.0 - a) * 255.0).round().astype(np.uint8)
    return arr[..., :3]

def save_batch_visualization1(input_batch, ground_truth_batch, output_batch, epoch, save_dir):
    batch_size = input_batch.shape[0]
    batch_save_dir = os.path.join(save_dir, f"epoch_{epoch}")
    os.makedirs(batch_save_dir, exist_ok=True)

    # ---- save each image (RGB uint8, no alpha, no gray veil)
    for idx in range(batch_size):
        inp = input_batch[idx].detach().cpu().permute(1, 2, 0).numpy()
        gt  = ground_truth_batch[idx].detach().cpu().permute(1, 2, 0).numpy()
        out = output_batch[idx].detach().cpu().permute(1, 2, 0).numpy()

        inp = _to_uint8_rgb(_denorm01(inp))
        gt  = _to_uint8_rgb(_denorm01(gt))
        out = _to_uint8_rgb(_denorm01(out))

        Image.fromarray(inp).save(os.path.join(batch_save_dir, f"input_{idx}.png"))
        Image.fromarray(gt).save(os.path.join(batch_save_dir, f"ground_truth_{idx}.png"))
        Image.fromarray(out).save(os.path.join(batch_save_dir, f"output_{idx}.png"))

    # ---- combined grid (denorm + white composition + white figure bg)
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size), facecolor="white")
    if batch_size == 1:
        axes = np.array([axes])  # make it 2D for uniform indexing

    for idx in range(batch_size):
        inp = input_batch[idx].detach().cpu().permute(1, 2, 0).numpy()
        gt  = ground_truth_batch[idx].detach().cpu().permute(1, 2, 0).numpy()
        out = output_batch[idx].detach().cpu().permute(1, 2, 0).numpy()

        inp = _composite_over_white(_denorm01(inp))
        gt  = _composite_over_white(_denorm01(gt))
        out = _composite_over_white(_denorm01(out))

        axes[idx, 0].imshow(inp);  axes[idx, 0].set_title("Input (Noisy)");    axes[idx, 0].axis("off")
        axes[idx, 1].imshow(gt);   axes[idx, 1].set_title("Ground Truth");     axes[idx, 1].axis("off")
        axes[idx, 2].imshow(out);  axes[idx, 2].set_title("Output (Denoised)");axes[idx, 2].axis("off")

    viz_path = os.path.join(batch_save_dir, f"visualization_batch_{epoch}.png")
    plt.savefig(viz_path, facecolor="white", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved visualization plot at {viz_path}")
def save_batch_visualization(input_batch, ground_truth_batch, output_batch, epoch, save_dir):
    """ Save each image in the batch individually and create a combined visualization plot. """

    batch_size = input_batch.shape[0]  # Get the batch size
    batch_save_dir = os.path.join(save_dir, f"epoch_{epoch}")
    os.makedirs(batch_save_dir, exist_ok=True)  # Ensure per-epoch directory exists

    for idx in range(batch_size):
        # Convert tensors to numpy
        input_img = input_batch[idx].detach().cpu().permute(1, 2, 0).numpy()
        ground_truth = ground_truth_batch[idx].detach().cpu().permute(1, 2, 0).numpy()
        output_img = output_batch[idx].detach().cpu().permute(1, 2, 0).numpy()

        # Reverse normalization if applied
        if input_img.max() < 1.1:
            input_img = (input_img * 0.5) + 0.5  # Convert from [-1,1] to [0,1]
            ground_truth = (ground_truth * 0.5) + 0.5
            output_img = (output_img * 0.5) + 0.5

        # Convert to [0, 255] and uint8
        input_img = (input_img * 255).astype(np.uint8)
        ground_truth = (ground_truth * 255).astype(np.uint8)
        output_img = (output_img * 255).astype(np.uint8)

        # Save images individually
        input_path = os.path.join(batch_save_dir, f"input_{idx}.png")
        gt_path = os.path.join(batch_save_dir, f"ground_truth_{idx}.png")
        output_path = os.path.join(batch_save_dir, f"output_{idx}.png")

        Image.fromarray(input_img).save(input_path)
        Image.fromarray(ground_truth).save(gt_path)
        Image.fromarray(output_img).save(output_path)

        print(f"Saved {input_path}")
        print(f"Saved {gt_path}")
        print(f"Saved {output_path}")

    # Save a combined visualization plot
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
    for idx in range(batch_size):
        axes[idx, 0].imshow(input_batch[idx].cpu().permute(1, 2, 0).numpy())
        axes[idx, 0].set_title("Input (Noisy)")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(ground_truth_batch[idx].cpu().permute(1, 2, 0).numpy())
        axes[idx, 1].set_title("Ground Truth")
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(output_batch[idx].cpu().permute(1, 2, 0).numpy())
        axes[idx, 2].set_title("Output (Denoised)")
        axes[idx, 2].axis("off")

    viz_path = os.path.join(batch_save_dir, f"visualization_batch_{epoch}.png")
    plt.savefig(viz_path)
    plt.close()
    print(f"Saved visualization plot at {viz_path}")
# without adding modified output images with input noisy images
def train_epoch_unet_womo(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for batch_idx, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        # Compute loss
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Training Loss: {epoch_loss / len(dataloader):.4f}")
    return epoch_loss / len(dataloader)

def validate_epoch_unet_womo(model, dataloader, criterion, epoch, save_dir, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()
            # Save visualizations only for the first batch of each epoch
            if batch_idx == 0:
                save_batch_visualization(images, targets, outputs, epoch,save_dir)
    print(f"Validation Loss: {epoch_loss / len(dataloader):.4f}")
    return epoch_loss / len(dataloader)

# Testing function

def test_unetsd(model, dataloader, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for idx, (input_images, _, input_filenames) in enumerate(dataloader):
            input_images = input_images.to(device)
            outputs = model(input_images)
            save_images(input_images, input_filenames, outputs, output_dir)
        print(f"Images Saved in the {output_dir}")

def test_unetsd_cluster(model, dataloader, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for idx, (input_images, input_filenames) in enumerate(dataloader):
            input_images = input_images.to(device)
            outputs = model(input_images)
            save_images(input_images, input_filenames, outputs, output_dir)
        print(f"Images Saved in the {output_dir}")

def save_images(input_images, input_filenames, outputs, output_dir):
    batch_size = input_images.shape[0]
    for idx in range(batch_size):
        output_img = outputs[idx].cpu().permute(1, 2, 0).numpy()
        output_img = (output_img * 255).astype(np.float32)
        output_img = np.clip(output_img, 0, 255).astype(np.uint8)
        # Ensure filename compatibility
        filename = input_filenames[idx]
        if isinstance(filename, (tuple, list)):
            filename = filename[0]
        output_path = os.path.join(output_dir, filename)
        # img = Image.fromarray(output_img)
        pil_arr, mode = _to_pil_ready(output_img)
        img = Image.fromarray(pil_arr if mode is None else pil_arr, mode=mode)

        img.save(output_path, format='PNG')  # Save as PNG for lossless quality


# Function to visualize input and ground truth images
def visualize_dataset(dataset, num_samples=5):

    for i in range(num_samples):
        input_image, ground_truth_image = dataset[i]
        input_image = input_image.permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC for visualization
        ground_truth_image = ground_truth_image.permute(1, 2, 0).numpy()

        plt.figure(figsize=(10, 5))

        # Plot input image
        plt.subplot(1, 2, 1)
        plt.imshow(input_image)
        plt.title("Input Image")
        plt.axis("off")

        # Plot ground truth image
        plt.subplot(1, 2, 2)
        plt.imshow(ground_truth_image)
        plt.title("Ground Truth Image")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


