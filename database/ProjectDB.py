import glob
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from itertools import chain

import numpy as np
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from database.db_quaries import *

logging.basicConfig(level="INFO")


def extract_name(file_path: str) -> tuple:
    """
    Extract platform, studySite, scene from filename. name from file path
    Parameters
    ----------
    file_path - string with file path

    Returns
    -------
    a tuple of strings with platform, studySite, scene.
    """

    # Parse parameters from file name
    platform, studySite, scene = file_path.split('/')[-3:]
    return platform, studySite, scene


class ProjectDB:
    """
    Class which interacts with the project database and retrieves queries
    """
    def __init__(self, db_path: str = "data/database.db", PS_dir: str = 'data/PS', S2_dir: str = 'data/S2', test_dir: str = 'data/test', val_dir: str = 'data/val'):
        """
        Initialisation functon which creates db or reads it in
        Parameters
        ----------
        db_path - path to database file
        PS_dir - path to where PS scenes are saved
        S2_dir - path to where S2 scenes are saved
        test_dir - path to where test polygons are saved
        val_dir - path to where validation polygons are saved
        """

        self.db_path = db_path
        self.PS_path = PS_dir
        self.S2_path = S2_dir
        self.test_dir = test_dir
        self.val_dir = val_dir

        # Create db if nto existing
        if not os.path.isfile(os.path.join(db_path)):
            logging.info(f'There is no such database file {db_path}.')
            try:
                self.creation()
                logging.info("Database was created!")
                logging.info(f'Go place your PS scenes into {PS_dir}.')
            except:
                raise ValueError(f"Database could not be created successfully.")
        else:
            # Connect to db if created
            self.conn = sqlite3.connect(db_path)
            self.cur = self.conn.cursor()

        # Create scene directories if they do not exist
        os.makedirs(S2_dir, exist_ok=True)
        os.makedirs(PS_dir, exist_ok=True)


    def creation(self):
        """
        Creates the database.
        """
        self.conn = sqlite3.connect(self.db_path)
        self.cur = self.conn.cursor()
        self.cur.executescript(CREATE_SCENE)
        self.cur.executescript(CREATE_MATCH)
        self.cur.executescript(CREATE_PATCH)
        self.conn.commit()

    @property
    def PS_scenes(self):
        """
        Gets a list of PS image names from the Scene table.

        Returns
        -------
        a list of PS images
        """
        self.cur.execute(GET_PS)
        return [x[0] for x in self.cur.fetchall()]

    @property
    def S2_scenes(self):
        """
        Gets a list of S2 image names from the Scene table.
        Returns
        -------
        a list of PS images

        """
        self.cur.execute(GET_S2)
        return [x[0] for x in self.cur.fetchall()]

    @property
    def sites(self):
        """
        Gets a list of study sites from the Scene table.
        Returns
        -------
        a lsit of study sites
        """
        self.cur.execute(GET_SCENES)
        return [x[0] for x in self.cur.fetchall()]

    @property
    def scenes(self):
        """
        Gets a list of scenes from the Scene table.
        Returns
        -------
        a list of scenes
        """
        self.cur.execute(GET_SITES)
        return [x[0] for x in self.cur.fetchall()]

    @property
    def match_IDs(self):
        """
        Gets a full list of match IDs from the Match table.
        Returns
        -------
        a list of macthIDs
        """
        self.cur.execute(GET_MATCH_IDS)
        return [x[0] for x in self.cur.fetchall()]

    @property
    def train_matches(self):
        """
        Gets a list of training matches with corresponding patch.

        Returns
        -------
        a list of tuples (matchID,
        """
        self.cur.execute(GET_TRAIN_TEST_MATCHES, (0,))
        return [x[0] for x in self.cur.fetchall()]

    @property
    def test_matches(self):
        """
        Gets a list of test matches with corresponding patch.
        Returns
        -------
        a list of tuples matchIDs
        """
        self.cur.execute(GET_TRAIN_TEST_MATCHES, (1,))
        return [x[0] for x in self.cur.fetchall()]

    @property
    def validate_matches(self):
        """
        Gets a list of validation matches with corresponding patch.
        Returns
        -------
        a list of tuples matchIDs
        """
        self.cur.execute(GET_TRAIN_TEST_MATCHES, (2,))
        return [x[0] for x in self.cur.fetchall()]

    def get_site_date(self, site):
        """
        Gets acquisition date for each study site.

        Parameters
        ----------
        site - a string of the study site

        Returns
        -------
        a string of the date in format YY-mm-dd
        """
        self.cur.execute(GET_SITE_DATE, (site,))
        return self.cur.fetchone()[0]

    def get_S2_from_PS(self, scene):
        """
        Gets corresponding SceneID of S2 for a given PS SceneID.

        Parameters
        ----------
        scene - a string with a PS SceneID

        Returns
        -------
        a SceneID of S2 which corresponds to the PS SceneID.
        """
        self.cur.execute(GET_S2_FROM_PS, (scene,))
        return self.cur.fetchone()[0]

    def get_site_crs(self, site, platform="PS"):
        """
        Gets a crs code for a given site.
        Parameters
        ----------
        site - a string with study site name
        platform - a string with a platform (PS or S2).

        Returns
        -------
        a string with crs code.
        """

        # Get CRS system
        self.cur.execute(GET_SITE_CRS, (site, platform,))
        crs = self.cur.fetchall()
        if len(crs) == 1:
            return crs[0][0]
        elif len(crs) == 0:
            raise ValueError("No CRS were returned, check StudySite name")
        else:
            raise ValueError('Too many CRS returned.')

    def get_study_site(self, scene):
        """
        Get study site for a given scene.

        Parameters
        ----------
        scene - a string with a SceneID

        Returns
        -------
        a string with a study site anme.
        """
        self.cur.execute(GET_STUDY_SITE, (scene,))
        return self.cur.fetchone()[0]

    def get_test_IDs_per_studysite(self, platform, study_site):
        """
        Gets patchIDs used fir testing for a given platform and study site.

        Parameters
        ----------
        platform - a string with a platform (PS or S2).
        study_site -  a string with a study site name

        Returns
        -------
        a list of patchIDs
        """
        GET_ID_STUDYSITE_TEST_replaced = GET_ID_STUDYSITE_TEST.replace("X", platform)
        self.cur.execute(GET_ID_STUDYSITE_TEST_replaced, (study_site,))
        return self.cur.fetchall()

    def get_study_site_params(self, site):
        """
        Gets crs, date start and end (+1 day), bounding box coordinates for a given study site.
        Parameters
        ----------
        site - a string with a study site name

        Returns
        -------
        a dictionary with crs, date stating and ending points and bounding box coordinates
        """
        date_start = self.get_site_date(site)
        date_end = str((datetime.strptime(date_start, '%Y-%m-%d') + timedelta(days=1)).date())

        params = {'site_crs': self.get_site_crs(site), 'date_start': date_start, 'date_end': date_end,
                  'bbox_coords': self.get_combined_bbox(site)}

        return params

    def get_platform(self, scene):
        """
        Gets platform for a given SceneID.

        Parameters
        ----------
        scene - a string with a SceneID.

        Returns
        -------
        a string with a platform value.
        """
        self.cur.execute(GET_PLATFORM, (scene,))
        return self.cur.fetchone()[0]

    def get_combined_bbox(self, studyArea, platform="PS"):
        """
        Returns min and max latitudes and longitudes to represent combined bounding box based on PS (or other) images.

        Parameters
        ----------
        studyArea - a string with a study area name
        platform - a string with a platform (PS or S2).

        Returns
        -------
        tuple (maxLat, maxLon, minLat, minLon)
        """
        # Get coordinates
        self.cur.execute(GET_COMBINED_BBOX, (studyArea, platform,))
        comb_bbox_coords = self.cur.fetchall()[0]
        return comb_bbox_coords

    def get_match_info(self, matchID):
        """
        Gets match information.

        Parameters
        ----------
        matchID -  a string with MatchID value.

        Returns
        -------
        a tuple with values:
                        PatchID_PS,
                        CentreLat_PS, CentreLon_PS,
                        PS_SceneID,
                        PatchID_S2,
                        CentreLat_S2, CentreLon_S2,
                        S2_SceneID
        """
        self.cur.execute(GET_MATCH_INFO, (matchID,))
        return self.cur.fetchone()

    def get_path(self, scene):
        """
        Gets path for a given SceneID

        Parameters
        ----------
        scene -  a string with a SceneID value.

        Returns
        -------
        relative path to the tif file.
        """
        if self.get_platform(scene) == "PS":
            return os.path.join(self.PS_path, self.get_study_site(scene), scene)
        else:
            return os.path.join(self.S2_path, self.get_study_site(scene), scene)

    def get_duplicate_patch_ids(self):
        """
        Gets ids of duplicate patch logs in the database.

        Returns
        -------
        a list of ids of duplicate patches.
        """
        self.cur.execute(GET_DUPLICATE_PATCH_IDS)
        return self.cur.fetchall()

    def deduplicate_patches(self):
        """
        Removes duplicate patches in the database.

        Returns
        -------
        None
        """
        for ids in self.get_duplicate_patch_ids():
            # Get a list of IDs of duplicates as integers
            ids = list(map(int, ids[0].split(',')))

            # Get query adjustments based on the amount of duplicates
            ids_query = ",".join("?" * (len(ids) - 1))

            try:
                with self.conn:
                    if ids[0] % 2 != 0:
                        self.cur.execute(REPLACE_DUPLICATE_PS_IDS.format(ids_query), ids)
                    else:
                        self.cur.execute(REPLACE_DUPLICATE_S2_IDS.format(ids_query), ids)
                    self.remove_patch(ids[1:])

            except sqlite3.Error as e:
                print(f"Error replacing duplicates with ids {ids}: {e}")

    def deduplicate_matches(self):
        """
        Removes duplicate matches in the database.
        Returns
        -------
        None
        """

        # Get ids of duplicates
        duplicateIDS = self.get_duplicate_matchIDs()

        # Remove all but the first one
        for matchIDs in duplicateIDS:
            matchIDs = matchIDs[1:]
            self.remove_matches(matchIDs)

    def remove_patch(self, patchIDs):
        """
        Removes patches with given PatchIDs

        Parameters
        ----------
        patchIDs -  a list of PatchIDs to be removed.

        Returns
        -------
        None
        """
        try:
            self.cur.execute(DELETE_PATCH_IDS.format(",".join("?" * len(patchIDs))), patchIDs)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error deleting duplicates: {e}")

    def remove_matches(self, matchIDs):
        """
        Removes matches with given MatchIDs
        Parameters
        ----------
        matchIDs - a list of matchIDs to be removed.

        Returns
        -------
        None
        """
        try:
            self.cur.execute(DELETE_MATCH_IDS.format(",".join("?" * len(matchIDs))), matchIDs)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error deleting duplicates: {e}")

    def get_duplicate_matchIDs(self):
        """
        Gets IDs of duplicate match logs.

        Returns
        -------
        a list of MatchIDs that are duplicated in the database.
        """
        try:
            self.cur.execute(GET_DUPLICATE_MATCH_IDS)
            duplicateIDS = self.cur.fetchall()
            return [list(map(int, i[0].split(','))) for i in duplicateIDS]

        except sqlite3.Error as e:
            print(f"Error deleting duplicate matches: {e}")

    def extract_scene_params(self, file_path: str) -> tuple:
        """
        Extracts parameters about given SceneID.

        Parameters
        ----------
        file_path - a path to the tif file.

        Returns
        -------
        a tuple consisting of parameters in order:
                            SceneID, Platform, Sensor, PxSize, AcqDate,
                            AcqTime, TopRightCornerLat, TopRightCornerLon,
                            BottomLeftCornerLat, BottomLeftCornerLon, StudySite
        """

        platform, studySite, scene = extract_name(file_path)

        if platform == 'PS':
            # Example: 20181031_095646_101b_3B_AnalyticMS_clip.tif
            name_list = scene.split('_')

            # Extract sensor
            sensor = name_list[(name_list.index("3B") - 1)]

            # Extract acquisition date and time
            acqDate = str(datetime.strptime(name_list[0], "%Y%m%d").date())
            acqTime = str(datetime.strptime(name_list[1], "%H%M%S").time())

        elif platform == 'S2':
            # Example: 20181031T101139_Accra_1.tif
            # Extract sensor
            sensor = platform

            # Extract acquisition date and time
            date_time = scene.split('.')[0].split("T")
            acqDate = str(datetime.strptime(date_time[0], "%Y%m%d").date())
            acqTime = str(datetime.strptime(date_time[1].split("_")[0], "%H%M%S").time())
        else:
            raise ValueError(f'{platform} is not a pre-defined platform')

        with rasterio.open(file_path) as src:
            # Get original coord sys
            original_crs = src.crs

            # Initialise reprojector to EPSG:4326
            transformer = Transformer.from_crs(original_crs, "EPSG:4326", always_xy=True)

            # Get bbox coord in EPSG:4326
            bounds = src.bounds
            TopRightCornerLon, TopRightCornerLat = transformer.transform(bounds.right, bounds.top)
            BottomLeftCornerLon, BottomLeftCornerLat = transformer.transform(bounds.left, bounds.bottom)
            # TopRightCornerLon, TopRightCornerLat = bounds.right, bounds.top
            # BottomLeftCornerLon, BottomLeftCornerLat = bounds.left, bounds.bottom

            # Get pixel size
            transform = src.transform
            if transform[0] == -transform[4]:
                pxSize = transform[0]
            else:
                raise ValueError(f'Pixels are not square for {scene}')

        # Collect parameters to be written into db
        params = (scene, platform, sensor, pxSize, acqDate, acqTime, TopRightCornerLat, TopRightCornerLon,
                  BottomLeftCornerLat, BottomLeftCornerLon, studySite, str(src.crs))
        return params

    def add_new_scenes(self, params):
        """
        Adds new scenes to the database.
        Parameters
        ----------
        params - a list of tuples, consisting of parameters in order:
                            SceneID, Platform, Sensor, PxSize, AcqDate,
                            AcqTime, TopRightCornerLat, TopRightCornerLon,
                            BottomLeftCornerLat, BottomLeftCornerLon, StudySite
        Returns
        -------
        None
        """
        try:
            # Insert new scenes
            self.cur.executemany(ADD_NEW_SCENE, params)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error inserting new scene: {e}")

    def add_new_patches(self, params):
        """
        Adds entries of new patches to the database
        Parameters
        ----------
        params - a list of tuples, consisting of parameters:
                            SceneID, x and y coordinates.

        Returns
        -------
         a list of uniqueKeys

        """

        try:
            # Insert new scenes
            self.cur.executemany(ADD_NEW_PATCH, params)
            self.conn.commit()

            # Get the PatchID
            self.cur.execute(GET_NEW_PATCH_ID, (len(params),))
            return [row[0] for row in self.cur.fetchall()]
        except sqlite3.Error as e:
            print(f"Error occurred inserting patches: {e}")

    def add_new_matches(self, matches):
        """
        Adds multiple matches to the database.

        Parameters
        ----------
        matches - list of tuples, each containing:
                        (PatchID_PS, PatchID_S2)

        Returns
        -------
        None
        """
        try:
            # Batch insert matches
            self.cur.executemany(ADD_NEW_MATCH, matches)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error inserting matches: {e}")

    def log_new_match(self, patches_params):
        """
        Logs the matches into the database.
        Parameters
        ----------
        patches_params - a list containing information about patches.
                                It assumes that odd element [i] is a patch in PS image,
                                and the even corresponding [i+i] element is
                                the same patch in S2 image.

        Returns
        -------
        None
        """

        patch_ids = self.add_new_patches(patches_params)
        patch_ids.sort()

        match_list = []

        for i in range(0, len(patch_ids), 2):
            match_list.append((patch_ids[i], patch_ids[i + 1]))

        self.add_new_matches(match_list)

    def log_new_scenes(self, scene_IDs: list[str]) -> None:
        """
        For non-exiting scenes (in the db), gets parameters and adds these missing scenes to the db.
        Parameters
        ----------
        scene_IDs -  a list not logged scenes.

        Returns
        -------
        None
        """

        params = []

        for sceneID in scene_IDs:
            params.append(self.extract_scene_params(sceneID))

        try:
            # Add parameters to db
            self.add_new_scenes(params)
        except Exception as e:
            print(f"Error processing: {e}")

    def deduplicate(self, radius: float==10) -> None:
        """
        Removes dupplicate patches and matches from the database.
        Parameters
        ----------
        radius - search radius along which to consider S2 patches as the same.

        Returns
        -------
        None
        """
        self.deduplicate_patches()
        self.deduplicate_matches()
        self.combine_S2_patches(radius=radius)

    def combine_S2_patches(self, radius: float) -> None:
        """
        Combines S2 patches within radius as the same and re-assigns patch IDs.

        Parameters
        ----------
        radius - search radius within which matches are considered the same in S2 scene.

        Returns
        -------
        None
        """
        def find_nearby_points(points, radius):
            tree = cKDTree(points)
            pairs = tree.query_pairs(radius)
            return pairs

        self.cur.execute(GET_S2_PATCHES)
        patches = self.cur.fetchall()

        for site in self.sites:
            site_patches = [patch for patch in patches if patch[-1] == site]
            coord = [[i[2], i[3]] for i in site_patches]

            points = np.array(coord)
            duplicates = find_nearby_points(points, radius)

            deleted = {}
            m = 0

            for i, j in sorted(duplicates):
                i = site_patches[i]
                j = site_patches[j]

                if i[1] == j[1]:
                    larger = j[0] if j[0] > i[0] else i[0]
                    smaller = j[0] if j[0] < i[0] else i[0]

                    if larger not in deleted and smaller not in deleted:
                        self.cur.execute('DELETE FROM Patch WHERE PatchID=?', (larger,))
                        self.cur.execute('UPDATE Match SET PatchID_S2 = ? WHERE PatchID_S2=?', (smaller, larger))
                        deleted[larger] = smaller
                        self.conn.commit()

                    else:
                        if larger in deleted:
                            self.cur.execute('DELETE FROM Patch WHERE PatchID=?', (smaller,))
                            self.cur.execute('UPDATE Match SET PatchID_S2 = ? WHERE PatchID_S2=?', (
                                deleted[larger], smaller))
                            deleted[smaller] = deleted[larger]
                        else:
                            self.cur.execute('DELETE FROM Patch WHERE PatchID=?', (larger,))
                            self.cur.execute('UPDATE Match SET PatchID_S2 = ? WHERE PatchID_S2=?', (
                                deleted[smaller], larger))
                            deleted[larger] = deleted[smaller]
                        self.conn.commit()

    def get_match_IDS(self, folder_dir: str) -> None:
        """
        Gets patch IDs that are within polygons in a given folder. Used for extracting test and validate patch IDs.
        Parameters
        ----------
        folder_dir - path to folder containing polygons in geojson format.

        Returns
        -------
        a kist of patch IDs in the polygons in a given folder.
        """
        polygon_paths = glob.glob(os.path.join(folder_dir, '*.geojson'))

        selected_IDS = []

        for polygon_file in polygon_paths:
            geom = json.load(open(polygon_file))['features'][0]['geometry']
            polygon = Polygon(list(chain(*geom['coordinates'])))

            study_site = os.path.splitext(os.path.basename(polygon_file))[0]
            points = self.get_patch_from_study_site_platform(study_site)

            for patchID, lat, lon in points:
                point = Point(lon, lat)
                if polygon.contains(point):
                    selected_IDS.append(patchID)

        return selected_IDS

    def update_match_TestTrainValidate(self) -> None:
        """
        Updates TestTrainValidate column in the Match table, based on data split.

        Returns
        -------
        None
        """
        test_IDs = self.get_match_IDS(self.test_dir)
        validate_IDs = self.get_match_IDS(self.val_dir)

        # https://stackoverflow.com/questions/2864842/common-elements-comparison-between-2-lists
        duplicates = set(test_IDs).intersection(validate_IDs)
        if len(duplicates) > 0:
            raise ValueError('There is overlap between test and validation areas.')
        else:
            self.update_test_sites(test_IDs, 1)
            self.update_test_sites(validate_IDs, 2)
            logging.info('Match table was updated with data split labels.')

    def update_test_sites(self, match_IDs, label):
        """
        Executes the SQL query to update TestTrainValidate column in the db.
        Parameters
        ----------
        match_IDs - list of match IDs
        label - what label to assign: 1 - for test, 2 - for validation, 0 - for train.

        Returns
        -------
        None

        """
        macth_ID_labeled = list(zip(match_IDs, [label] * len(match_IDs)))
        self.cur.executemany(UPDATE_TEST_SITES, macth_ID_labeled)
        self.conn.commit()

    def get_patch_from_study_site_platform(self, site: str, platform: str="S2") -> list[tuple[int, float, float]]:
        """
        Gets a list of all patches (ID, and coordinates) from a given study site and platform.
        Parameters
        ----------
        site - a string with a study site name.
        platform - a string with a platform name (PS or S2).

        Returns
        -------
        a list of the patches (ID, and coordinates) from a given study site and platform.
        """
        self.cur.execute(GET_S2_COORD_STUDYSITE, (site, platform,))
        return self.cur.fetchall()

    def update(self, studySite: str = "*") -> None:
        """
        Updates the database with new files in the data_dir.

        Parameters
        ----------
        studySite - a string with a study site name.

        Returns
        -------
        None
        """

        # Gather all tif files
        existing_scenes_PS = [i for i in glob.glob(f'{self.PS_path}/{studySite}/*.tif') if "udm" not in i]
        existing_scenes_S2 = [i for i in glob.glob(f'{self.S2_path}*/{studySite}/*.tif')]
        existing_scenes = existing_scenes_PS + existing_scenes_S2

        # Create empty list contained for missing scenes
        missing_scenes = []

        # Get logged scenes
        logged_scenes = self.scenes

        # Iterate through all tifs and add missing ones
        for scene_file in existing_scenes:
            _, _, scene = extract_name(file_path=scene_file)

            # Check if scene is in db, if not add it to missing list
            if scene not in logged_scenes:
                missing_scenes.append(scene_file)

        # Batch add new scenes
        self.log_new_scenes(missing_scenes)

    def close(self):
        """Closes the database connection and cursor."""
        self.conn.commit()
        self.cur.close()
        self.conn.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Closes database on exit"""
        self.close()

    def get_match_status(self, PS_IDs: int, S2_IDs: int) -> list[tuple[float,  float, float , float, int]]:
        """
        Returns coordinates of both patches in the match and if the patches are matching - 1 or not - 0.
        Parameters
        ----------
        PS_IDs - PS patch IDs
        S2_IDs - S2 patch IDs

        Returns
        -------
        a list with PS and S2 coorinates plus the match label (1 or 0).
        """
        info = []

        for PS_ID, S2_ID in zip(PS_IDs, S2_IDs):
            self.cur.execute(GET_MATCH_STATUS, (S2_ID, PS_ID))
            info.append(self.cur.fetchone())
        return info

    def get_time_difference(self, site: str, mode: str='test') -> list[tuple[int, float, float, int, float]]:
        """
        Gets acquisition time differences for match pair.
        Parameters
        ----------
        site - a string with a study site name.
        mode - data split mode: test, train or validate.

        Returns
        -------
        a list of the acquisition time differences for match pair plus the infor about the match (PS patch ID,
            PS patch latitude and longitude, S2 patch ID, time differnece) .
        """
        mode = 0 if mode == 'train' else 1 if mode == 'test' else 2

        self.cur.execute(GET_TIME_DIFFERENCES, (mode, site,))
        return self.cur.fetchall()

    def get_points_in_radius(self, PS_lat: float, PS_lon: float, radius: float, study_site: str, plat: str="S2", mode: str='test') -> list[tuple[float, float]]:
        """
        Gets S2 patches within radius of a g iven PS patch (by providing it's coordinates).
        Parameters
        ----------
        PS_lat - PS patch latitude
        PS_lon - PS patch longitude
        radius - search radius
        study_site - study site name
        plat - a string with a platform name (PS or S2).
        mode - 'train' or 'test'

        Returns
        -------
        a list of retirved S2 patches within certain radius of a given PS patch.
        """
        mode = 0 if mode == 'train' else 1 if mode == 'test' else 2
        params = (mode, study_site, plat, PS_lat, PS_lat, PS_lon, PS_lon, radius)

        self.cur.execute(GET_POINT_WITHIN_RADIUS, params)
        S2_IDs = self.cur.fetchall()

        return S2_IDs
