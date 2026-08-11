import argparse
import logging
import os
import random

import lightning.pytorch as pl
import numpy as np
import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from training.datamodule.ssl_dataset import DebrisDataModuleSSL
from training.ssl_pre_training import PatchMatchSSL
from config.configs import Configs
from utils.setups import set_up_logger
from utils.checks import check_if_trained, check_if_pretrained
from utils.extra import cleanup

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("medium")

"""Script to pre-train PS and S2 encoders using self-supervised SimCLR to refine SSL4EO weights"""


def main(config_path='config/setup_f_3_2.json', wandb_project_name=None, fast_dev=False):
    # Set seeds
    seed_everything(1121218)
    random.seed(1121218)
    np.random.seed(1121218)
    torch.manual_seed(1121218)
    torch.use_deterministic_algorithms(True)

    # Get configurations
    config = Configs(config_path)

    # Stop if model is (pre)trained for config
    if not fast_dev:
        if check_if_trained(config):
            exit()
        elif check_if_pretrained(config):
            exit()

    # Change wandb name if given
    if wandb_project_name:
        config.project_name = wandb_project_name

    # Get SSL transformations
    left_transforms, right_transforms = config.get_transforms_strategy()

    # Get ssl modes (PS and S2)
    for ssl_mode in config.ssl_mode:
        wandb_logger = None if fast_dev else set_up_logger(config, ssl=True)

        # Get encoders
        PS_encoder, S2_encoder, backbone = config.set_up_models(branch=ssl_mode)

        # Set up models
        model = PatchMatchSSL(left_encoder=PS_encoder, right_encoder=S2_encoder, shared_bacbone=backbone, ssl_mode=ssl_mode,
                              hidden_dim=config.ssl_hidden_dim, lr=config.ssl_lr,
                              temperature=config.ssl_temp, weight_decay=config.ssl_weight_decay,
                              max_epochs=config.ssl_epochs)

        # Set up dataloader for SSL training
        SSL_datamodule = DebrisDataModuleSSL(paths=config.get_ssl_pairs(), mode=ssl_mode,
                                             batch_size=config.ssl_batch_size,
                                             contrast_transforms=left_transforms,
                                             right_transforms=right_transforms,
                                             tile_size=config.tile_size,
                                             right_tile_size=config.right_tile_size,
                                             normalise=config.normalise, bandwise=config.bandwise,
                                             num_workers=16)

        if fast_dev:
            trainer = pl.Trainer(fast_dev_run=True)
        else:
            checkpoint_callback = ModelCheckpoint(
                dirpath=config.checkpoint_dir,
                filename=f'{config.run_name}_{ssl_mode}' + '-SSL-{val_acc_top3:.2f}-{epoch:02d}',
                monitor='validate_acc_top3',
                mode='max',
                save_weights_only=True
            )

            lr_monitor = LearningRateMonitor('epoch')

            trainer = pl.Trainer(
                devices=1,
                accelerator='gpu',
                max_epochs=config.ssl_epochs,
                logger=wandb_logger,
                callbacks=[lr_monitor, checkpoint_callback],
                log_every_n_steps=1
            )

        # Train
        trainer.fit(model, datamodule=SSL_datamodule)

        wandb_logger.experiment.finish()
        cleanup()

    if not fast_dev:
        config.ssl_pretrained = True
        config.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='A path to setup configuration file, e.g.: "config/setup_1.json"')
    parser.add_argument('wandb_project_name', type=str, nargs='?', help='Which wandb project to use: predefined in condifguration file or other, e.g.: PatchMatch_trial_runs')
    parser.add_argument("--fast_dev", action="store_true", help="Use fast developer debugger mode.")

    args = parser.parse_args()

    if os.path.exists(args.config_file):
        main(args.config_file, wandb_project_name=args.wandb_project_name, fast_dev=args.fast_dev)
    else:
        parser.error("Such configuration file does not exist")
