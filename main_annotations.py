import subprocess
import logging
import os
import glob
import argparse
import platform

from utils.extra import build_pyramids
from data_preparation.annotations.get_ndi import save_ndi_raster
from data_preparation.annotations.annotation_data import log_endpoints_from_json
from database import ProjectDB
from config.configs import Configs

logging.basicConfig(level=logging.INFO)

"""
This script creates files (ndi, empty geojson) for annotation process and loads them into
automatically created QGIS projects. This script might not work on different OS systems (e.g., MacOS).
There is a need to have QGIS, GDAL installed  pip install GDAL==3.8.4.
"""


def main(config_path='config/proposed.json'):
    # Get configurations
    config = Configs(config_path)

    db = ProjectDB.ProjectDB(config.DB_path, config.PS_dir, config.S2_dir, config.test_poly_dir, config.validate_poly_dir)

    if not config.ndi_calculated:
        for scene in db.scenes:
            # Get scene parameters
            path = db.get_path(scene)
            site = db.get_study_site(scene)
            plat = db.get_platform(scene)

            # Create output path
            out_path = os.path.join(config.ndi_dir, site, "ndi_" + scene)

            # Calculate ndi if not done already
            if not os.path.isfile(out_path):
                save_ndi_raster(path=path, out_path=out_path, platform=plat)

                # Build pyramids
                build_pyramids(path)
                logging.info(f'NDI for {scene} scene was calculated.')
        config.ndi_calculated = True

    # Create QGIS projects
    if not config.qgis_projects_created:
        if platform.system() == "Darwin":
            python_exe = config.python_dir
            env = os.environ.copy()
            env["PROJ_LIB"] = "/Applications/QGIS.app/Contents/Resources/proj"
        else:
            python_exe = "python"
            env = None
        for PS_path in db.PS_scenes:
            # There are memory issues thus it has to run as a subprocess
            logging.info(f"Processing {PS_path} in a separate process.")
            subprocess.run([python_exe, "data_preparation/annotations/create_qgis_project.py", PS_path, config_path], env=env)
        config.qgis_projects_created = True

    # Extract annotations after user input
    response = input(f'Go to you qgis projects and create the annotations.\nWhen you are done press enter or x to exit: ')
    if response.upper() != 'X':
        # Extract annotations
        path = os.path.join(config.annotations_dir, '*', '*.geojson')
        geojson_files = glob.glob(path)
        geojson_files.sort()
        if len(geojson_files) != 0:
            for file in geojson_files:
                log_endpoints_from_json(file, db)
            db.deduplicate(radius=config.S2_search_radius)
            logging.info(f"All annotations were extracted to the database.")

    db.close()
    config.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='A path to setup configuration file, e.g.: "config/setup_1.json"')

    args = parser.parse_args()

    if os.path.exists(args.config_file):
        main(args.config_file)
    else:
        parser.error("Such configuration file does not exist")
