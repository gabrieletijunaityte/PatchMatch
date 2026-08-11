import argparse
import random
import torch
from lightning import seed_everything

from testing.classification_metrics import calculate_classification_performance
from testing.retrieval_metrics import *

from config.configs import Configs
from utils.setups import setup_model, setup_dataloader

"""
Script to evaluate the trained model as defined in configuration file for the MD patch matching task.
Does the evaluation for global, local and radius-bound scopes. For the latter also executes the optimal 
drift-radius search.
"""

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("medium")


def main(config_path='config/proposed.json', weight_path=None, figures=False):
    # Set seeds
    seed_everything(1121218)
    random.seed(1121218)
    np.random.seed(1121218)
    torch.manual_seed(1121218)
    torch.use_deterministic_algorithms(True)

    # Get configurations
    config = Configs(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else "cpu")

    calculate_classification_performance(config, device, acc_mode='validate_top_5_acc')

    while calculate_global_top_k(config):
        # If enters here, obtain prediction matrix first
        model = setup_model(config, testing_mode=True, acc_mode='validate_top_5_acc').to(device)

        dataloader = setup_dataloader(config)
        dataloader.setup('test')
        test_dataloader = dataloader.test_dataloader()

        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                model.predict_step(batch, batch_idx, save_path=config.pred_path)

    # Get top k accuracies per study sites
    calculate_local_top_k(config)

    # Get predicted pair maps of top-1 predictions per local scope (for visualisations)
    save_local_top_1_pairs(config)

    # Get top k accuracies within search radius
    calculate_drift_top_k(config, speeds=[0.02, 0.04, 0.1, 0.2, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.5, 2], site='Accra_2')
    calculate_drift_top_k(config, speeds=[0.02, 0.04, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0], site='Marmara')

    if figures:
        config.normalise = False
        get_top_k_images(config, k=3, plot=True, save=True)
        get_top_k_images(config, k=3, plot=True, save=True)
        get_top_k_images(config, k=3, PS_patch_ID='135', plot=True, save=True)
        get_top_k_images(config, k=3, PS_patch_ID='5829', plot=True, save=True)
        get_top_k_images(config, k=3, plot=True, save=True)
        get_top_k_images(config, k=3, plot=True, save=True)
        get_top_k_images(config, k=3, plot=True, save=True)
        get_top_k_images(config, k=3, plot=True, save=True)
        get_top_k_images(config, k=3, plot=True, save=True)
        get_top_k_images(config, k=3, plot=True, save=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='A path to setup configuration file, e.g.: "config/setup_1.json"')

    parser.add_argument('weights_file', type=str, nargs='?', help='A path to the model weights, e.g.: "output/weights/setup_1-epoch=20-val_loss=0.00.ckpt."')

    parser.add_argument('--figures', action="store_true", help="Generate random pair figures with predictions")

    args = parser.parse_args()

    if os.path.exists(args.config_file):
        main(args.config_file, args.weights_file, figures=args.figures)
    else:
        parser.error("Such configuration file does not exist")
