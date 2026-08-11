import json
import logging
import math
import os

import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from database.ProjectDB import ProjectDB

logging.basicConfig(level=logging.INFO)


class TilePreProcessing:
    """
    A class to generate tiles from Marine Debris patches annotations.
    """

    def __init__(self, db: ProjectDB, tile_size: int, template: str, save_dir: str = 'intermediate/tiles/',
                 resample: bool = True, S2_res: int = 10, PS_res: int = 3):
        """
        A constructor to generate TilePreProcessing class.
        :param db: a database object.
        :param tile_size: a desired tile size for PS images and S2 if not resampled
        :param template: a filename template
        :param save_dir: a directory to save generated tiles. Defaults to intermediate/tiles/
        :param resample: whether to resample S2 images or not. Defaults to True.
        :param S2_res: resolution of S2 images. Defaults to 10.
        :param PS_res: resolution of PS images. Defaults to 3.
        """
        self.db = db
        self.img_paths = db.scenes
        self.tile_size_PS = tile_size
        self.template = template
        self.save_dir = save_dir
        self.S2_res = S2_res
        self.PS_res = PS_res
        self.resample = resample
        if resample:
            self.tile_size_S2 = tile_size
        else:
            self.tile_size_S2 = int(tile_size / (S2_res / PS_res))

        # Create save_dir folder if it does not exist
        os.makedirs(save_dir, exist_ok=True)

    def generate_tile(self, scene: str, lon: float, lat: float, out_path: str, size: int = 300,
                      resample: bool = False) -> None:
        """
        Saves a tile centered on lon and lat, by cropping scene to the tile_size parameter. If needed it bilinearly
        upsamples the tile.
        Sources:
        https://gis.stackexchange.com/questions/367832/using-rasterio-to-crop-image-using-pixel-coordinates-instead-of-geographic-coord
        https://stackoverflow.com/questions/75499270/print-the-pixel-coordinates-of-a-specific-point-in-the-raster
        https://rasterio.readthedocs.io/en/stable/topics/resampling.html
        :param scene: a string of a scene name
        :param lon: centre longitude of annotation
        :param lat: centre latitude of annotation
        :param out_path: output path
        :param size: size of tile
        :param resample: a boolean indicating, if S2 image needs to be resampled
        :return: None
        """
        # Get full path to the image
        path = self.db.get_path(scene)

        if resample:
            resample_fact = self.S2_res / self.PS_res
        else:
            resample_fact = 1

        # Get tile size specific to each platform
        org_tile_size = math.ceil(size * resample_fact ** (-1))

        with (rasterio.open(path) as src):
            # Get pixelwise coordinates of annotation
            row, col = src.index(lon, lat)

            # Top left corner coordinate
            col_off = max(0, min(col - org_tile_size // 2, src.width - org_tile_size))
            row_off = max(0, min(row - org_tile_size // 2, src.height - org_tile_size))

            window = Window(col_off, row_off, org_tile_size, org_tile_size)
            transform = src.window_transform(window) * src.transform.scale(1 / resample_fact, 1 / resample_fact)

            # Create a new cropped raster
            profile = src.profile
            profile.update({
                'height': size,
                'width': size,
                'dtype': src.dtypes[0],
                'driver': 'GTiff',
                'transform': transform})

            # Read and resample the data
            data = src.read(
                window=window,
                out_shape=(src.count, size, size),
                resampling=Resampling.bilinear
            )

            # Extract band names
            band_descriptions = [src.descriptions[i] for i in range(src.meta['count'])]

            with rasterio.open(out_path, 'w', **profile) as dst:
                for i in range(data.shape[0]):
                    dst.write(data[i], i + 1)
                    dst.set_band_description(i + 1, band_descriptions[i])

    def generate_tile_pairs(self) -> dict[int: list]:
        """
        Generates tile pairs for all match annotations in the database.
        :return: a dictionary of lists of MatchID tile filenames.
        """
        pair_dic = {}
        for match_ID in self.db.match_IDs:
            pair_dic[match_ID] = self.generate_tile_pair(match_ID)

        return pair_dic

    def generate_tile_pair(self, match_ID: int) -> list[str]:
        """
        Generates individual tile pair for a specified matchID.
        :param match_ID: matchID value.
        :return: a list of tile filenames.
        """
        # Get match ID info
        PS_ID, PS_lat, PS_lon, PS_scene, S2_ID, S2_lat, S2_lon, S2_scene = self.db.get_match_info(match_ID)

        PS_name = self.template.replace('?', '{}').format(PS_ID, 'PS')
        S2_name = self.template.replace('?', '{}').format(S2_ID, 'S2')

        PS_path = os.path.join(self.save_dir, PS_name)
        S2_path = os.path.join(self.save_dir, S2_name)

        if not os.path.isfile(PS_path):
            self.generate_tile(PS_scene, lon=PS_lon, lat=PS_lat, out_path=PS_path, size=self.tile_size_PS,
                               resample=False)

        if not os.path.isfile(S2_path):
            self.generate_tile(S2_scene, lon=S2_lon, lat=S2_lat, out_path=S2_path, size=self.tile_size_S2,
                               resample=self.resample)
        logging.info(f'Tiles are available for the match: {match_ID}.')

        return [PS_path, S2_path]


def test_tiles_json(pairs, out_path):
    """Create all possible test tile mapping (true matches and all negative pairs in the test data split)"""
    all_PS = []
    all_S2 = []

    matches = {}
    all_tiles = []

    # Add all matches to all_tiles
    for _, tile_lst in pairs.items():
        PS_tile = tile_lst[0]
        S2_tile = tile_lst[-1]

        if PS_tile not in all_PS:
            all_PS.append(PS_tile)
        if S2_tile not in all_S2:
            all_S2.append(S2_tile)

        matches[(PS_tile, S2_tile)] = 1
        all_tiles.append([PS_tile, S2_tile, 1])

    # Add all non-matches to all_tiles
    for PS_tile in all_PS:
        for S2_tile in all_S2:
            if (PS_tile, S2_tile) not in matches:
                all_tiles.append([PS_tile, S2_tile, 0])

    # Save to jeojson
    with open(out_path, "w") as all_tiles_json:
        json.dump(all_tiles, all_tiles_json)


def save_ssl_pairs_file(config, idx_train, idx_validate, pair_dic):
    """Saves PS and S2 tiles based on data split"""
    if os.path.exists(config.ssl_pairs):
        return

    ssl_pairs = {}
    ssl_pairs['PS'] = {'train': [pair_dic[key][0] for key in idx_train],
                       'validate': [pair_dic[key][0] for key in idx_validate]}
    ssl_pairs['S2'] = {'train': list(set([pair_dic[key][1] for key in idx_train])),
                       'validate': list(set([pair_dic[key][1] for key in idx_validate]))}

    with open(config.ssl_pairs, "w") as pairs_json:
        json.dump(ssl_pairs, pairs_json)

    logging.info(f"SSL tile information saved into: {config.ssl_pairs}")
