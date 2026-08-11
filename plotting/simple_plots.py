import glob
import logging
import os
import math

import matplotlib.pyplot as plt
import torch


def stackRGB(B, G, R, tensor):
    """Creates rgb stakc from a given tensor and the rgb channel indices"""
    blue = torch.select(tensor, 0, B)
    green = torch.select(tensor, 0, G)
    red = torch.select(tensor, 0, R)

    tile_rgb = torch.stack((red, green, blue))
    return tile_rgb

def plot_match(tensor_a, tensor_b, match, prediction, loss=None, similarity=None, label_a=None, label_b=None, show=True, save_dir=None, normalise=None):
    """Plot PS and S2 match with their prediction, similarity, label info"""
    # Normalize tensors
    if not normalise:
        tensor_a = (tensor_a - tensor_a.min()) / (tensor_a.max() - tensor_a.min())
        tensor_b = (tensor_b - tensor_b.min()) / (tensor_b.max() - tensor_b.min())

    # Reshape tensors
    img_a = tensor_a.squeeze().permute(1, 2, 0).cpu().numpy()
    img_b = tensor_b.squeeze().permute(1, 2, 0).cpu().numpy()

    # Initialise figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Set border colour based on the label correctness
    border_color = "green" if match == 1 else "red"

    # https://stackoverflow.com/questions/68018852/how-to-change-border-width-in-matplotlib
    for axis in ['top', 'bottom', 'left', 'right']:
        axes[0].spines[axis].set_linewidth(2.5)
        axes[0].spines[axis].set_color("green")
        axes[1].spines[axis].set_linewidth(2.5)
        axes[1].spines[axis].set_color(border_color)

    # Plot tensor_a
    axes[0].imshow(img_a)
    axes[0].set_title(label_a or "PS tile", fontsize=12)

    # Plot tensor_b
    axes[1].imshow(img_b)
    axes[1].set_title(label_b or "S2 tile", fontsize=12)

    match_status = 'Match' if match == 1 else 'Non-match'
    match_color = "green" if match == prediction else "red"

    # Set the title
    fig.suptitle(f"Predicted Correctly a {match_status}"
                 if match == prediction
                 else f"Predicted Wrongly, should have been a {match_status}", fontsize=16, c=match_color)

    if similarity is not None:
        fig.text(0.5, 0.005, f"With similarity: {similarity} and loss: {loss}", ha='center', fontsize=12)
    plt.tight_layout()

    axes[0].set_yticklabels([])
    axes[0].set_xticklabels([])
    axes[1].set_yticklabels([])
    axes[1].set_xticklabels([])
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    if show:
        plt.show()
    else:
        i = len(glob.glob(os.path.join(save_dir, 'prediction_*.png')))
        title = os.path.join(save_dir, f'prediction_{i}.png')
        plt.savefig(title)
        logging.info(f'Figure saved to {title}.')
        plt.close()

def visualise_augmented_pair(tensor_a, tensor_b, show=True, save_fn=None):
    """Visualises a pair of two augmented views of the same image"""
    if tensor_a.shape[0] == 13:
        b, g, r = 1, 2, 3
        title = 'Sentinel-2'
    else:
        b, g, r = 0, 1, 2
        title = 'PlanetScope'

    # Convert to rgb stack
    tensor_a = stackRGB(b, g, r, tensor_a)
    tensor_b = stackRGB(b, g, r, tensor_b)

    # Normalise
    min_val = min(tensor_a.min().item(), tensor_b.min().item())
    max_val = max(tensor_a.max().item(), tensor_b.max().item())
    tensor_a = (tensor_a - min_val) / (max_val - min_val)
    tensor_b = (tensor_b - min_val) / (max_val - min_val)

    # Reshape for visualisation
    img_a = tensor_a.squeeze().permute(1, 2, 0).cpu().numpy()
    img_b = tensor_b.squeeze().permute(1, 2, 0).cpu().numpy()

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot tensor_a
    axes[0].imshow(img_a)
    axes[0].set_title("Tile view 1", fontsize=12)
    axes[0].axis('off')

    # Plot tensor_b
    axes[1].imshow(img_b)
    axes[1].set_title("Tile view 2", fontsize=12)
    axes[1].axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(save_fn)
        logging.info(f'Figure saved to {save_fn}.')
        plt.close()

def plot_imgs(imgs_list):
    """Plot a list of images"""
    fig, axes = plt.subplots(1, len(imgs_list), figsize=(10, 5))
    for i, img in enumerate(imgs_list):
        if img.shape[0] == 4:
            img = stackRGB(0, 1, 2, img)
        elif img.shape[0] == 13:
            img = stackRGB(1, 2, 3, img)

        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        axes[i].imshow(img)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def plot_imgs_grid(imgs_list, title=None):
    """Plot a list of images into a grid"""

    # Get num of rows and columns
    cols = min(4, len(imgs_list))
    rows = max(1, math.ceil(len(imgs_list) / cols))

    # Initialise figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    # Plot each image in a list
    for i, img in enumerate(imgs_list):
        if img.shape[0] == 4:
            img = stackRGB(0, 1, 2, img)
        elif img.shape[0] == 13:
            img = stackRGB(1, 2, 3, img)

        img = img.permute(1, 2, 0).cpu().numpy()
        # Normalise based on the first image
        if i == 0:
            min_val = img.min()
            max_val = img.max()
        img = (img - min_val) / (max_val - min_val)

        axes[i].imshow(img)
        axes[i].axis("off")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title)

    plt.tight_layout()
    plt.show()