import json
import torch
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt

def unet_transformation(width=224, height=224):
    return transforms.Compose([
        transforms.Resize((width, height)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

def collate_fn(batch):
    batch = [item for sublist in batch for item in sublist]
    images, labels = zip(*batch)
    max_length = max(label.size(0) for label in labels)
    padded_labels = [torch.cat([label, torch.zeros(max_length - label.size(0))]) for label in labels]
    images = torch.stack(images, dim=0)
    labels = torch.stack(padded_labels, dim=0)
    return images, labels

def save_loss_plot(train_losses, val_losses, plot_dir, plot_name, len_train, len_val):
    plt.figure()
    plt.plot(range(1, len_train + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len_val + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(plot_dir, plot_name))
    plt.close()

def save_loss_plot1( val_losses, plot_dir, plot_name, len_val):
    plt.figure()
    plt.plot(range(1, len_val + 1), val_losses, label='training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.savefig(os.path.join(plot_dir, plot_name))
    plt.close()

def save_error_plot(maes, plot_dir, plot_name, plot_title ='Mean Absolute Error' ):
    plt.figure()
    plt.plot(range(1, len(maes) + 1), maes)
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title(plot_title)
    plt.savefig(os.path.join(plot_dir, plot_name))
    plt.close()

def save_args_to_json(args, dir_name, filename='args_file.json'):
    # Save args params
    print("Parameters:")
    print(args)
    filename = os.path.join(dir_name, filename)
    args_dict = vars(args)
    with open(filename, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)
        print("Arguments parameters saved to args_file.")

def visualize_crops(cropped_image, i, left_x, mid_x, right_x):
    plt.figure(figsize=(6, 6))
    if isinstance(cropped_image, torch.Tensor):
        cropped_image = cropped_image.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
        plt.imshow(cropped_image.numpy())  # Convert tensor to numpy for plotting
    else:
        plt.imshow(cropped_image)
    plt.title(f'Crop {i + 1} (Left: {left_x}, Mid: {mid_x}, Right: {right_x})')
    plt.axis('off')
    plt.show()

def visualize_crops_dht(cropped_image, gt_coords, i, left_x, right_x):
    plt.figure(figsize=(6, 6))
    if isinstance(cropped_image, torch.Tensor):
        cropped_image = cropped_image.permute(1, 2, 0)
        plt.imshow(cropped_image.numpy())
    else:
        plt.imshow(cropped_image)
    for box in gt_coords:
        y1, x1, y2, x2 = box.tolist()
        plt.scatter(x1, y1, color='red', marker='o')
        plt.scatter(x2, y2, color='blue', marker='o')
    plt.title(f'Crop {i} (Left: {left_x}, Right: {right_x})')
    plt.axis('off')
    plt.show()


def load_json(json_path):
    with open(json_path, "r") as file:
        return json.load(file)

def parse_key(filename: str):
    """
    Return a normalized key for pairing:
        (image_id:int, crop_id:int, code:str|None)
    Accepts both 'image_3_crop_2_UTnXwc.png' and 'image_3_UTnXwc_crop_2.png'.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split('_')

    image_id, crop_id, code = None, None, None

    # find image id and crop id wherever they appear
    for i, p in enumerate(parts):
        if p == 'image' and i + 1 < len(parts) and parts[i+1].isdigit():
            image_id = int(parts[i+1])
        if p == 'crop' and i + 1 < len(parts) and parts[i+1].isdigit():
            crop_id = int(parts[i+1])

    # find an alphanumeric token that is not 'image'/'crop' or a pure number
    # (your random tag like UTnXwc, w06Cr, etc.)
    for p in parts:
        if p not in {'image', 'crop'} and not p.isdigit():
            code = p
            break

    return (image_id, crop_id, code)


