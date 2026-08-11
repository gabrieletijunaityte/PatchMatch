import random

import torchvision.transforms.v2.functional as F
from torchvision.transforms import v2
import torch

"""Functions for augmentations"""

def shuffle_bands(img: torch.tensor):
    bands = torch.randperm(img.shape[0])
    return img[bands, :, :]


def adjust_brightness(img: torch.tensor, level: str, s2=False):
    if s2:
        brightness_factor = random.uniform(0.6, 1.4)
    else:
        brightness_factor = random.uniform(0.8, 1.2)
    img_ = img.clone()
    for band in range(img.size(0)):
        if level == "mid":
            img_[band] = F.adjust_brightness(img[band].unsqueeze(0), brightness_factor)
        else:
            if s2:
                img_[band] = F.adjust_brightness(img[band].unsqueeze(0), brightness_factor + random.uniform(-0.2, 0.2))
            else:
                img_[band] = F.adjust_brightness(img[band].unsqueeze(0), brightness_factor + random.uniform(-0.05, 0.05))
    return img_

def adjust_contrast(img: torch.tensor):
    contrast_factor = random.uniform(0.5, 3)
    img_ = img.clone()
    for band in range(img.size(0)):
        img_[band] = F.adjust_contrast(img[band].unsqueeze(0), contrast_factor)
    return img_

def random_zoom_effect(img: torch.tensor, min_val, max_val):
    org_size = img.shape[1]
    size = torch.randint(min_val, max_val, (1,)).item()

    img_ = img.clone()
    img_ = v2.Resize(size=size, interpolation=v2.InterpolationMode.BILINEAR)(img_)
    img_ = v2.CenterCrop(min(img_.shape[1], org_size))(img_)
    img_ = v2.Resize(org_size, interpolation=v2.InterpolationMode.BILINEAR)(img_)

    return img_

