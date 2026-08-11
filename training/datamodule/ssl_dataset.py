import torch
from lightning import pytorch as pl
from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

from data_preparation.data_inspection import load_min_max
from training.datamodule.dataset import seed_worker
from utils.extra import open_file_as_tensor, normalisation


class DebrisDatasetSSL(Dataset):
    """Dataset for SSL (single-modality"""
    def __init__(self, paths: list[str], platform: str = None, transforms: v2 = None, tile_size: int = 256,
                 normalise: bool = True, bandwise: bool = True):
        # Paths
        self.paths = paths

        # Normalisation
        self.normalise = normalise
        if self.normalise:
            self.min_val, self.max_val = load_min_max(f'output/data_inspection/{platform}_stats_01_99.csv',bandwise=bandwise)

        # Tile size
        self.tile_size = tile_size

        # Augmentations
        self.base_transforms = v2.CenterCrop(size=(self.tile_size, self.tile_size))
        self.transforms = transforms
        self.ssl_view_transformations = SSLTransformations(base_transforms=self.base_transforms,
                                                           contrastive_transforms=self.transforms,
                                                           n_views=2)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Get image's path
        path = self.paths[idx]

        # Open as a tensor
        img_tensor = open_file_as_tensor(path)

        # Normalisation
        if self.normalise:
            img_tensor = normalisation(img_tensor, self.min_val, self.max_val)
        else:
            img_tensor = img_tensor / 10000

        # Generate augmented views
        img_tensors = self.ssl_view_transformations(img_tensor)

        return img_tensors


class SSLTransformations:
    """Transformations for SimCLR SSL"""
    # https://lightning.ai/docs/pytorch/LTS/notebooks/course_UvA-DL/13-contrastive-learning.html
    def __init__(self, base_transforms: v2, contrastive_transforms: v2 = None, n_views: int = 2) -> None:
        self.base_transforms = base_transforms
        self.contrastive_transforms = v2.Compose([contrastive_transforms, base_transforms])
        self.n_views = n_views

    def __call__(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [self.contrastive_transforms(x) for i in range(self.n_views)]


class SequentialLoader:
    """ Sequential dataloader, for pre-training shared between platforms architectures"""
    def __init__(self, dataloaders: {str: DataLoader}):
        self.dataloaders = dataloaders
        self.current_side = 'left'
        self._reset_iterators()

    def _reset_iterators(self):
        self.iterable_dataloaders = {side: iter(d) for side, d in self.dataloaders.items()}

    def __len__(self):
        return sum(len(d) for d in self.dataloaders.values())

    def __iter__(self):
        self._reset_iterators()
        return self

    def __next__(self):
        self._check_exhaustion()

        side = self._check_side_exhaustion()

        try:
            batch = next(self.iterable_dataloaders[side])
            self.current_side = 'left' if side != 'left' else 'right'
        except StopIteration:
            self.iterable_dataloaders.pop(side)
            self._check_exhaustion()

            if self.iterable_dataloaders:
                side = self._check_side_exhaustion()
                batch = next(self.iterable_dataloaders[side])
            else:
                raise StopIteration

        return side, batch

    def _check_exhaustion(self):
        if not self.iterable_dataloaders:
            raise StopIteration

    def _check_side_exhaustion(self):
        if self.current_side not in self.iterable_dataloaders:
            side = 'left' if self.current_side != 'left' else 'right'
        else:
            side = self.current_side
        return side


class DebrisDataModuleSSL(pl.LightningDataModule):
    """SSL dataloaders for SimCLR pre-training"""
    def __init__(self, paths: dict[str: list[str]], mode: str, batch_size: int = 32,
                 contrast_transforms: v2 = None, contrast_transforms_right: v2 = None, tile_size: int = 256,
                 right_tile_size: int = None,
                 normalise: bool = True, bandwise: bool = True, num_workers: int = 16, **kwargs):
        super().__init__()

        # Paths map for train, validation and testing
        self.paths = paths

        # Batch size
        self.batch_size = batch_size

        # Transformations
        self.contrast_transforms = contrast_transforms
        self.contrast_transforms_right = contrast_transforms_right or contrast_transforms

        # Pre-training mode
        self.mode = mode if mode in ['PS', 'S2', "combined", 'sequential'] else NotImplementedError

        # Tile size information
        self.tile_size = tile_size
        self.right_tile_size = right_tile_size or tile_size

        # Normalisation
        self.normalise = normalise
        self.bandwise = bandwise

        self.num_workers = num_workers
        self.g = torch.Generator().manual_seed(0)


    def setup(self, stage: str):
        if stage == "fit":
            if self.mode in ['PS', 'S2']:
                transforms = self.contrast_transforms if self.mode == 'PS' else self.contrast_transforms_right

                tile_size = self.tile_size if self.mode == 'PS' else self.right_tile_size
                self.train_data = DebrisDatasetSSL(paths=self.paths[self.mode]['train'], platform=self.mode,
                                                   transforms=transforms, tile_size=tile_size,
                                                   normalise=self.normalise, bandwise=self.bandwise)

                self.validate_data = DebrisDatasetSSL(paths=self.paths[self.mode]['validate'], platform=self.mode,
                                                      transforms=transforms, tile_size=tile_size,
                                                      normalise=self.normalise, bandwise=self.bandwise)

            elif self.mode in ['sequential', 'combined']:
                self.PS_train_dataset = DebrisDatasetSSL(paths=self.paths['PS']['train'], platform='PS',
                                                         transforms=self.contrast_transforms, tile_size=self.tile_size,
                                                         normalise=self.normalise, bandwise=self.bandwise)

                self.S2_train_dataset = DebrisDatasetSSL(paths=self.paths['S2']['train'], platform='S2',
                                                         transforms=self.contrast_transforms_right,
                                                         tile_size=self.right_tile_size,
                                                         normalise=self.normalise, bandwise=self.bandwise)

                self.PS_validate_dataset = DebrisDatasetSSL(paths=self.paths['PS']['validate'], platform='PS',
                                                            transforms=self.contrast_transforms,
                                                            tile_size=self.tile_size,
                                                            normalise=self.normalise, bandwise=self.bandwise)

                self.S2_validate_dataset = DebrisDatasetSSL(paths=self.paths['S2']['validate'], platform='S2',
                                                            transforms=self.contrast_transforms_right,
                                                            tile_size=self.right_tile_size,
                                                            normalise=self.normalise, bandwise=self.bandwise)

    def train_dataloader(self):
        return self._dataloader('train')

    def val_dataloader(self):
        return self._dataloader('validate')

    def _dataloader(self, val_train):
        if self.mode in ['combined', 'sequential']:
            if val_train == 'validate':
                PS_dataset = self.PS_validate_dataset
                S2_dataset = self.S2_validate_dataset
            elif val_train == 'train':
                PS_dataset = self.PS_train_dataset
                S2_dataset = self.S2_train_dataset
            else:
                raise NotImplementedError

            PS_loader = DataLoader(PS_dataset, self.batch_size, shuffle=True,
                                   num_workers=self.num_workers, worker_init_fn=seed_worker,
                                   generator=self.g, drop_last=True, pin_memory=False,
                                   persistent_workers=(val_train == 'validate'))
            S2_loader = DataLoader(S2_dataset, self.batch_size, shuffle=True,
                                   num_workers=self.num_workers, worker_init_fn=seed_worker,
                                   generator=self.g, drop_last=True, pin_memory=True,
                                   persistent_workers=(val_train == 'validate'))
            if self.mode == 'combined':
                return CombinedLoader({"left": PS_loader, "right": S2_loader}, mode="max_size")
            else:
                return SequentialLoader({'left': PS_loader, 'right': S2_loader})
        else:
            dataset = self.validate_data if val_train == 'validate' else self.train_data
            return DataLoader(dataset, self.batch_size, shuffle=True, num_workers=self.num_workers,
                              worker_init_fn=seed_worker, generator=self.g, drop_last=True,
                              pin_memory=(val_train == 'validate'),
                              persistent_workers=(val_train == 'validate'),
                              )
