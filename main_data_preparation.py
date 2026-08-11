import argparse
import glob
from natsort import natsorted

from data_preparation.data_inspection import save_no_data_inspection, save_pixel_counts, save_band_stats
from data_preparation.tile_preprocessing import *
from data_preparation.tile_preprocessing import save_ssl_pairs_file
from database import ProjectDB
from config.configs import Configs

logging.basicConfig(level=logging.INFO)

"""
This script creates tiles out of annotation list, based on provided in config file parameters.
In addition it calculates statistics for all tiles.
"""


def main(config_path='config/proposed.json', stats=True):
    # Get configurations
    config = Configs(config_path)
    paths = config.get_train_pairs()

    if not paths or (config.ssl_pairs_file is not None):
        db = ProjectDB.ProjectDB(config.DB_path, config.PS_dir, config.S2_dir, config.test_poly_dir, config.validate_poly_dir)

        # Generate tiles
        tiles = TilePreProcessing(db, config.pt_size, config.pt_template, config.pt_dir, resample=config.resample)
        pair_dic = tiles.generate_tile_pairs()
        logging.info(f'All pre-tiles generated.')

        db.update_match_TestTrainValidate()

        # Save tile train/validate/test dictionary as json
        idx_train = db.train_matches
        idx_validate = db.validate_matches
        idx_test = db.test_matches

        # Make up tile data split files
        pairs = {'train': {key: pair_dic[key] for key in idx_train},
                 'validate': {key: pair_dic[key] for key in idx_validate},
                 'test': {key: pair_dic[key] for key in idx_test}}

        with open(config.pt_json_path, "w") as pairs_json:
            json.dump(pairs, pairs_json)
        logging.info(f"Pre tile information saved into: {config.pt_json_path}")

        if config.ssl_pairs_file:
            save_ssl_pairs_file(config, idx_train, idx_validate, pair_dic)

        # Save all possible pairs of test tiles
        test_tiles_json(pairs['test'], config.pt_test_json_path)
        logging.info(f"Test data information saved into: {config.pt_test_json_path}")

        db.close()

    # Getting statistics of data
    if stats:
        stat_dir = 'output/data_inspection'
        os.makedirs(stat_dir, exist_ok=True)

        for mode in ['train', 'test', 'validate']:
            # No Data Counting
            save_no_data_inspection(paths, mode, stat_dir)

            # Calculate mean, std, min and max values
            if mode == 'train':
                # Get pixel counts
                save_pixel_counts(paths, mode, stat_dir)

                S2_bands = glob.glob(os.path.join(stat_dir, f'*S2_{mode}_*.csv'))
                PS_bands = glob.glob(os.path.join(stat_dir, f'*PS_{mode}_*.csv'))

                S2_bands = natsorted(S2_bands)
                PS_bands = natsorted(PS_bands)

                save_band_stats(PS_bands, 'PS', 0.025, 0.975, stat_dir)
                save_band_stats(S2_bands, 'S2', 0.025, 0.975, stat_dir)
                save_band_stats(PS_bands, 'PS', 0.01, 0.99, stat_dir)
                save_band_stats(S2_bands, 'S2', 0.01, 0.99, stat_dir)
                save_band_stats(PS_bands, 'PS', 0, 1, stat_dir)
                save_band_stats(S2_bands, 'S2', 0, 1, stat_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='A path to setup configuration file, e.g.: "config/setup_1.json"')

    parser.add_argument('--stats', action="store_true", help="Calculate statistics for tiles (pixel counts, min, max values).")

    args = parser.parse_args()

    if os.path.exists(args.config_file):
        main(args.config_file, stats=args.stats)
    else:
        parser.error("Such configuration file does not exist")
