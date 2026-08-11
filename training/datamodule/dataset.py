import random
import logging

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import v2

from data_preparation.data_inspection import load_min_max
from utils.extra import open_file_as_tensor, normalisation


def seed_worker(worker_id):
    """Seed worker for reproducibility."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class TrainingAugmentations:
    """Augmentation strategy implementation"""
    def __init__(self, base_transforms: v2, transforms: v2 = None, p: float=0.5) -> None:
        self.base_transforms = base_transforms
        self.transforms = transforms or None
        self.p = p

    def __call__(self, x: torch.Tensor, p) -> list[torch.Tensor]:
        if p > self.p and (self.transforms is not None):
            x = self.transforms(x)
            return self.base_transforms(x)
        else:
            return self.base_transforms(x)


class DebrisDataset(Dataset):
    """Dataset for model training"""

    # https://stackoverflow.com/questions/306400/how-can-i-randomly-select-choose-an-item-from-a-list-get-a-random-element
    # https://discuss.pytorch.org/t/dataloader-for-a-siamese-model-with-concatdataset/66085/4
    # https://www.datacamp.com/tutorial/pytorch-lightning-tutorial?dc_referrer=https%3A%2F%2Fapp.datacamp.com%2F
    def __init__(self, positive_pairs: dict[int: list[str]], positive_prob: float = 0.5, tile_size: int = 256,
                 trans_prob: float = 0.5, trans: v2 = None, right_tile_size: int = None, right_trans: v2 = None,
                 normalise=True, bandwise=True, shuffle=True):

        # Map of positive pairs
        self.positive_pairs = positive_pairs

        # MatchIDs
        self.match_IDs = list(positive_pairs.keys())

        # Should elements be shuffled
        if shuffle:
            match_IDs = list(positive_pairs.keys())
            random.shuffle(match_IDs)
            logging.info('Shuffle on')
            self.match_IDs = match_IDs
        else:
            logging.info('Shuffle off')

        # Probability of returning positive pair
        self.positive_prob = positive_prob

        # Tile sizing information
        self.tile_size = tile_size
        self.right_tile_size = right_tile_size if right_tile_size else tile_size

        # Transformation information
        self.trans = TrainingAugmentations(
            base_transforms=v2.CenterCrop(size=(tile_size, tile_size)),
            transforms=trans,
            p=trans_prob
        )
        self.right_trans = TrainingAugmentations(
            base_transforms=v2.CenterCrop(size=(right_tile_size, right_tile_size)),
            transforms= right_trans or trans,
            p=trans_prob
        )

        # Normalisation information
        self.normalise = normalise
        if self.normalise:
            self.left_min, self.left_max = load_min_max('output/data_inspection/PS_stats_025_975.csv', bandwise=bandwise)
            self.right_min, self.right_max = load_min_max('output/data_inspection/S2_stats_025_975.csv', bandwise=bandwise)

    def __len__(self):
        return len(self.match_IDs)

    def __getitem__(self, idx):

        # Get match ID
        match_ID = self.match_IDs[idx]

        # Sample positive/negative pair
        p = random.random()
        positive_pair = p <= self.positive_prob

        # Read in paths for positive pair
        path_left = self.positive_pairs[match_ID][0]
        path_right = self.positive_pairs[match_ID][1]

        if positive_pair:
            match = 1
        else:
            # If negative pair, find a path replacement
            match = 0
            while True:
                random_ID = random.choice(self.match_IDs)
                random_path = self.positive_pairs[random_ID][1]
                if random_path != path_right:
                    path_right = random_path
                    break

        # Load in tensors
        tensor_left = open_file_as_tensor(path_left)
        tensor_right = open_file_as_tensor(path_right)

        if self.normalise:
            tensor_left = normalisation(tensor_left, self.left_min, self.left_max)
            tensor_right = normalisation(tensor_right, self.right_min, self.right_max)
        else:
            # Scale back by multiplying by the scale factor
            tensor_left = tensor_left / 10000
            tensor_right = tensor_right / 10000

        # Randomly apply transformations
        tensor_left = self.trans(tensor_left, p=random.random())
        tensor_right = self.right_trans(tensor_right, p=random.random())

        # Return ids
        right_id = int(path_right.split('_')[-2])
        left_id = int(path_left.split('_')[-2])

        return tensor_left.to(torch.float32), tensor_right.to(torch.float32), match, right_id, left_id


class DebrisDataModule(LightningDataModule):
    def __init__(self, pairs: dict[str: dict[str: list[
        str]]], tile_size: object, right_tile_size: int = None, train_positive_prob: float = 0.5, test_validate_positive_prob: float = 0.5, trans_prob: float = 0.5, batch_size: int = 32, val_batch_size: int = None, test_batch_size: int = None, trans: v2 = None, right_trans: v2 = None, num_workers: int = 16, normalise: object = True, bandwise: object = True, **kwargs: object) -> None:
        super().__init__()

        # Data splits
        self.train_pairs = pairs['train']
        self.validate_pairs = pairs['validate']
        self.test_pairs = pairs['test']

        # Tile sizes
        self.tile_size = tile_size
        self.right_tile_size = right_tile_size

        # Batch positive:negative proportions
        self.train_positive_prob = train_positive_prob
        self.test_validate_positive_prob = test_validate_positive_prob

        # Batch sizes
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        # Augmentations
        self.trans_prob = trans_prob
        self.trans = trans
        self.right_trans = right_trans

        # Normalisation
        self.normalise = normalise
        self.bandwise = bandwise

        self.num_workers = num_workers
        self.g = torch.Generator().manual_seed(0)


    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            print("Train dataset")
            self.train_data = DebrisDataset(positive_pairs=self.train_pairs,
                                            positive_prob=self.train_positive_prob,
                                            tile_size=self.tile_size, right_tile_size=self.right_tile_size,
                                            trans_prob=self.trans_prob, trans=self.trans,
                                            right_trans=self.right_trans, shuffle=False,
                                            normalise=self.normalise, bandwise=self.bandwise)

            print("Validate dataset")
            self.val_data = DebrisDataset(positive_pairs=self.validate_pairs,
                                          positive_prob=self.test_validate_positive_prob,
                                          tile_size=self.tile_size, right_tile_size=self.right_tile_size,
                                          trans_prob=1,
                                          trans=None, right_trans=None, shuffle=False,
                                          normalise=self.normalise, bandwise=self.bandwise)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            print("Test dataset")
            self.test_data = DebrisDataset(positive_pairs=self.test_pairs,
                                           positive_prob=self.test_validate_positive_prob,
                                           tile_size=self.tile_size, right_tile_size=self.right_tile_size,
                                           trans_prob=1,
                                           trans=None, right_trans=None, shuffle=False,
                                           normalise=self.normalise, bandwise=self.bandwise)

    def train_dataloader(self):
        return DataLoader(self.train_data, self.batch_size, shuffle=True,
                          num_workers=self.num_workers,
                          prefetch_factor=2,
                          worker_init_fn=seed_worker, generator=self.g, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, self.val_batch_size, shuffle=False,
                          persistent_workers=True, num_workers=self.num_workers,
                          worker_init_fn=seed_worker, generator=self.g, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, self.test_batch_size, shuffle=False,
                          persistent_workers=True, num_workers=self.num_workers,
                          worker_init_fn=seed_worker, generator=self.g, pin_memory=True)