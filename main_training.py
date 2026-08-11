import argparse
import logging
import os
import random

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import wandb

from testing.classification_metrics import save_results
from config.configs import Configs
from utils.setups import setup_model, setup_dataloader, set_up_logger
from utils.checks import check_if_trained
from utils.extra import cleanup

"""
This script trains a model based on a provided configuration file.
The metrics are logged and can be monitored with wandb.
"""

# For running things on mac, this changes nothing for linux
mp.set_start_method('fork', force=True)

wandb.finish()
cleanup()

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("medium")


def main(config_path='config/propsoed.json', wandb_project_name=None, fast_dev=False):
    # Set seeds
    seed_everything(1121218)
    random.seed(1121218)
    np.random.seed(1121218)
    torch.manual_seed(1121218)
    torch.use_deterministic_algorithms(True)

    # Get configurations
    config = Configs(config_path)

    # Stop if model is trained for config
    if check_if_trained(config) and not fast_dev:
        exit()

    # Change wandb name if given
    if wandb_project_name:
        config.project_name = wandb_project_name

    # Set up the logger
    wandb_logger = None if fast_dev else set_up_logger(config)

    # Load dataloader for training
    dataloader = setup_dataloader(config)

    # Set up the model
    model = setup_model(config, testing_mode=False)

    # Set up trainer for quick testing
    if fast_dev:
        trainer = pl.Trainer(fast_dev_run=True)
        trainer.fit(model, datamodule=dataloader)
        exit()

    # Define the ModelCheckpoint
    top_k_checkpoints = {
        f'model_{k}': ModelCheckpoint(dirpath=config.checkpoint_dir, filename=f'{config.run_name}-{{validate_top_{k}_acc:.2f}}-{{epoch:02d}}', monitor=f'validate_top_{k}_acc', mode='max', save_weights_only=True)
        for k in [1, 3, 5, 10]}

    # Define the callback that saves the learning rate used per epoch
    lr_monitor = LearningRateMonitor('epoch')

    trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=config.epochs, logger=wandb_logger, callbacks=[
                                                                                                                    lr_monitor] + list(top_k_checkpoints.values()), log_every_n_steps=1, # accumulate_grad_batches=16
    )

    # Train the model
    try:
        trainer.fit(model, datamodule=dataloader)
    except RuntimeError as e:
        print(e)
        if "out of memory" in str(e).lower():
            print("Out of Memory. Going for testing")
            torch.cuda.empty_cache()
        else:
            raise e
    finally:
        dataloader.setup('test')
        device = torch.device("cuda" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else "cpu")
        model = setup_model(config, testing_mode=True, acc_mode='validate_top_5_acc').to(device)
        results = trainer.test(model, dataloaders=dataloader.test_dataloader())[0]

    # Load results if available
    out_fp = 'output/model_performance_validate_top_5_acc.csv'
    results_df = pd.read_csv(out_fp) if os.path.exists(out_fp) else pd.DataFrame()

    # Check if results are already logged
    if config.run_name not in results_df['run_name'].tolist():
        logging.info(f'The results of {config.run_name} are already saved.')
        exit()

    save_results(config, out_fp, results, results_df)

    del model, dataloader
    cleanup()


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
