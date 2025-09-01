import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import torchvision.transforms as transforms


def save_visualization(input_img, ground_truth, output_img, epoch, save_dir):
    """ Function to save input, ground truth, and output images """
    input_img = input_img[0].detach().cpu().permute(1, 2, 0).numpy()  # Convert tensor to numpy
    ground_truth = ground_truth[0].detach().cpu().permute(1, 2, 0).numpy()
    output_img = output_img[0].detach().cpu().permute(1, 2, 0).numpy()

    # Reverse normalization if applied
    if input_img.max() < 1.1:  # Check if normalization was applied
        input_img = (input_img * 0.5) + 0.5  # Convert from [-1,1] to [0,1]
        ground_truth = (ground_truth * 0.5) + 0.5
        output_img = (output_img * 0.5) + 0.5

    # Plot images
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(input_img)
    ax[0].set_title("Input (Noisy)")
    ax[0].axis("off")

    ax[1].imshow(ground_truth)
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")

    ax[2].imshow(output_img)
    ax[2].set_title("Output (Denoised)")
    ax[2].axis("off")

    # Save the plot
    save_path = os.path.join(save_dir, f"epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved visualization for epoch {epoch} at {save_path}")


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
        img = Image.fromarray(output_img)
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


