import argparse
import logging
import os

import ee

from data_preparation.obtain_S2 import get_S2_images
from data_preparation.prepare_PS import get_PS_TOARef
from database import ProjectDB
from config.configs import Configs
from utils.extra import get_PS_paths

logging.basicConfig(level=logging.INFO)

"""
This script pre-processes (from TOARad to TOARef) and logs PS scenes into a database.
Then, based on PS scene date and bounding box coordinates, it downloads S2 scenes from Google Earth Engine. 
Scenes are made available in the user's Google drive and have to be manually downloaded and placed in corresponding folders.
Once the S2 scenes are manually downloaded, they are also logged into the database.
"""


def main(config_path='config/proposed.json'):
    # Get configurations
    config = Configs(config_path)

    db = ProjectDB.ProjectDB(config.DB_path, config.PS_dir, config.S2_dir, config.test_poly_dir, config.validate_poly_dir)

    # Covert all PS scenes from TOARad to TOARef
    if not config.PS_converted:
        PS_paths = get_PS_paths(config.PS_dir)
        for path in PS_paths:
            get_PS_TOARef(path)

        # Update db from existing files
        db.update()
        config.PS_converted = True

    # Get corresponding S2 images
    if not config.S2_available:
        if not ee.Authenticate():
            ee.Authenticate()
        ee.Initialize(project=config.GEE_project)

        for site in db.sites:
            params = db.get_study_site_params(site)
            get_S2_images(site, params, config.S2_dir)
        logging.info(f'Go to your Google drive and download newly exported images.')

        # Update database from user input
        response = input(f'Go to your google drive and download S2 scenes into {config.S2_dir} directory.\nWhen you are done press enter or x to exit: ')
    else:
        response = input(f'Press enter to update the database or x to exit: ')

    if response.upper() != 'X':
        db.update()
        logging.info("Database was updated.")
        config.S2_available = True

    config.save()
    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='A path to setup configuration file, e.g.: "config/setup_1.json"')

    args = parser.parse_args()

    if os.path.exists(args.config_file):
        main(args.config_file)
    else:
        parser.error("Such configuration file does not exist")
