import argparse
import os
import random
import logging
import gc

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
from lightning import seed_everything
import torch

from testing.top_k_metrics import calculate_top_k_accuracy, get_top_k_images
from utils.configs import Configs, load_model, load_test_dataloader

logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("medium")


def main(config_path, weight_path=None):
    # Set seeds
    seed_everything(1121218)
    random.seed(1121218)
    np.random.seed(1121218)
    torch.manual_seed(1121218)
    torch.use_deterministic_algorithms(True)

    # Get configurations
    config = Configs(config_path)

    while calculate_top_k_accuracy(config):
        model = load_model(config, pre_trained=True, acc_mode='validate_top_3_acc')
        test_loader = load_test_dataloader(config, config.batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                model.predict_step(batch, batch_idx, save=True, save_path=config.pred_path)

                del batch
                torch.cuda.empty_cache()
                gc.collect()

    # config.normalise=False
    #
    # get_top_k_images(config, k=8, plot=True, save=True, path=os.path.join(config.pred_dir, f'{config.run_name}_top_8_pred_1.png'))
    # get_top_k_images(config, k=8, plot=True, save=True, path=os.path.join(config.pred_dir, f'{config.run_name}_top_8_pred_2.png'))
    # get_top_k_images(config, k=8, img_name='intermediate/tiles/unresampled/Patch_135_PS.tif', plot=True, save=True, path=os.path.join(config.pred_dir, f'{config.run_name}_top_8_pred_3.png'))
    # get_top_k_images(config, k=8, img_name='intermediate/tiles/unresampled/Patch_5829_PS.tif', plot=True, save=True, path=os.path.join(config.pred_dir, f'{config.run_name}_top_8_pred_4.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file',
                        type=str,
                        help='A path to setup configuration file, e.g.: "config/setup_1.json"')

    parser.add_argument('weights_file',
                        type=str, nargs='?',
                        help='A path to the model weights, e.g.: "output/weights/setup_1-epoch=20-val_loss=0.00.ckpt."')

    args = parser.parse_args()

    if os.path.exists(args.config_file):
        main(args.config_file, args.weights_file)
    else:
        parser.error("Such configuration file does not exist")
