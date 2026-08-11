import json
import logging
import os

import geopandas as gpd

from database.ProjectDB import ProjectDB
from config.configs import Configs

logging.basicConfig(level=logging.INFO)


def dummy_annotation_file(geojson_path: str, site_crs: str) -> None:
    """
    Function creates an annotation geojson file based on PS scene. It contains one dummy entry.
    Inspired: https://stackoverflow.com/questions/16920700/building-a-geojson-with-python

    Parameters
    ----------
    geojson_path - a path with filename to where geojson will be written.
    site_crs - the coordinate system of geojson.

    Returns
    -------
    None
    """

    # Create an empty feature list
    features = []

    # Create a dummy line
    fake_feature = {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}}

    # Add it to features
    features.append(fake_feature)

    # Create a json for feature collection
    feature_collection = {"type": "FeatureCollection", "features": features, "crs": {"properties": {"name": site_crs}}}

    # Save the geojson
    with open(geojson_path, 'w') as f:
        json.dump(feature_collection, f)

    logging.info(f'Annotation file created: {geojson_path}')

    return


def dummy_polygon_file(site_crs: str, geojson_path: str) -> None:
    """
    Creates a geojson file for test/validation polygons with a dummy polygon.

    Parameters
    ----------
    site_crs -  the coordinate system of geojson.
    geojson_path - a path with filename to where geojson will be written.

    Returns
    -------

    """
    # Create an empty feature list
    features = []

    # Create a dummy polygon
    fake_feature = {"type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}}

    # Add it to features
    features.append(fake_feature)

    # Create a json for feature collection
    feature_collection = {"type": "FeatureCollection", "features": features, "crs": {"properties": {"name": site_crs}}}

    # Save the geojson
    with open(geojson_path, 'w') as f:
        json.dump(feature_collection, f)

    logging.info(f'Annotation file created: {geojson_path}')


def parse_endpoint_coords(annotations: gpd.GeoSeries) -> list[tuple]:
    """
    Return the beginning and end point coordinates from a geopandas geoseries of linestring features.
    Sources: https://stackoverflow.com/questions/20474549/extract-points-coordinates-from-a-polygon-in-shapely

    Parameters
    ----------
    annotations - a geoseries of linestring features.

    Returns
    -------
     a list of tuples where:
                the first element is a tuple of the beginning point coordinates (lon, lat) and
                the second element is a tuple of the end point coordinates (lon, lat).
    """
    # Extract x and y coordinates for both platforms
    coordinates = [((geom.coords.xy[0][0], geom.coords.xy[-1][0]), (geom.coords.xy[0][1], geom.coords.xy[-1][1])) for
                   geom in annotations.geometry]
    return coordinates


def log_endpoints_from_json(path: str, db: ProjectDB) -> None:
    """
    Updates db with patches and their matches annotations from a geojson file.

    Parameters
    ----------
    path - a path to geojson file containing match annotations as line features.
    db - a database object.

    Returns
    -------
    None
    """

    # Get PS scene name from file name
    PS_scene = os.path.splitext(os.path.basename(path))[0] + '.tif'

    # Get S2 scene name from PS scene:
    S2_scene = db.get_S2_from_PS(PS_scene)

    # Open geojson path
    annotations = gpd.read_file(path)

    # Extract PS and S2 (lat, lon) coordinates from gpd dataframe
    coords = parse_endpoint_coords(annotations)

    # Create an empty patch list
    patch_list = []

    # Iterate through annotated points pairs
    # Ignore the first dummy annotation
    for coord in coords[1:]:
        patch_list.append((PS_scene, coord[0][1], coord[0][0]))
        patch_list.append((S2_scene, coord[1][1], coord[1][0]))

    db.log_new_match(patch_list)
    logging.info(f"Annotations from {os.path.basename(path)} were extracted to the database.")

    return


def save_points_to_json(tuple_list: list[tuple], site_crs: str, geojson_path: str) -> None:
    """
    Creates geojson file with points from a given tuple.
    Parameters
    ----------
    tuple_list - a list of tuples with point coordinates.
    site_crs - the coordinate system of geojson.
    geojson_path - a path with filename to where geojson will be written.

    Returns
    -------
    None
    """

    features = []

    for tupl in tuple_list:
        # Create a point
        point = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [tupl[1], tupl[0]]}}

        # Add it to features
        features.append(point)

    # Create a json for feature collection
    feature_collection = {"type": "FeatureCollection", "features": features, "crs": {"properties": {"name": site_crs}}}

    # Save the geojson
    with open(geojson_path, 'w') as f:
        json.dump(feature_collection, f)


def save_mode_points_to_json(db: ProjectDB, mode: str = 'test') -> None:
    """
    Parameters
    ----------
    db - annotation db object
    mode - for which mode to save points. "test" or "validation" only.

    Returns
    -------
    None
    """
    if mode == 'train':
        raise NotImplementedError("Points cannot be extracted for train mode.")

    # Possible test. validation
    PS = {"Marmara": [], 'Accra_2': [], 'Accra_3': [], 'Accra_1': []}
    S2 = {"Marmara": [], 'Accra_2': [], 'Accra_3': [], 'Accra_1': []}

    # Get matched points for mode
    matches = db.validate_matches if mode == 'validate' else db.test_matches

    # Sort out points into sites
    for i in matches:
        match_info = db.get_match_info(i)

        if 'Marmara' in match_info[-1]:
            key = "Marmara"
        elif "Accra_3" in match_info[-1]:
            key = "Accra_3"
        elif "Accra_2" in match_info[-1]:
            key = "Accra_2"
        elif "Accra_1" in match_info[-1]:
            key = "Accra_1"

        PS[key].append((match_info[1], match_info[2]))
        S2[key].append((match_info[-3], match_info[-2]))

    # Prepare site list for saving files
    sites = ['Marmara']
    if mode == 'validate':
        sites.append('Accra_3')
    elif mode == 'train':
        sites.append('Accra_1')
    else:
        sites.append('Accra_2')

    path = 'intermediate/platform_points'
    os.makedirs(path, exist_ok=True)

    # Save points in geojsons
    for site in sites:
        crs = db.get_site_crs(site)
        save_points_to_json(S2[site], crs, f"{path}/S2_{site}_{mode}.json")
        save_points_to_json(PS[site], crs, f"{path}/PS_{site}_{mode}.json")

if __name__ == '__main__':
    # TODO: test this
    os.chdir('../..')

    config_path = 'config/proposed.json'
    config = Configs(config_path)

    db = ProjectDB(config.DB_path, config.PS_dir, config.S2_dir, config.test_poly_dir, config.validate_poly_dir)

    save_mode_points_to_json(db, mode='validate')
    save_mode_points_to_json(db, mode='test')