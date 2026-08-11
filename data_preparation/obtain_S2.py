import time
import os
import glob

import logging

import geopandas as gpd
from shapely.geometry import Polygon
import geemap
import ee

logging.basicConfig(level=logging.INFO)

"""Scripts to download S2 images from Google Earth ENgine"""

def export_img(img: ee.Image, folder_name: str, file_name: str, roi_ee: ee.FeatureCollection) -> None:
    """
    Exports image into Google Drive.
    Source: https://github.com/csaybar/EEwPython/blob/master/10_Export.ipynb
    :param img: GEE image object to be exported
    :param folder_name: a string of study site, which becomes a folder name
    :param file_name: a string for the image name
    :param roi_ee: an ee.Feature object of studySite bbox
    :return: None
    """

    # Get image region
    region = roi_ee.geometry().bounds()

    logging.info(f"Preparing scene {file_name}...")

    # Export image to Google Drive
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=file_name,
        folder=folder_name,
        fileNamePrefix=file_name,
        region=region.getInfo()['coordinates'],
        scale=10,
        maxPixels=1e13
    )

    # Start the task
    task.start()
    logging.info(f"Exporting {file_name} to Google Drive {folder_name} folder...")
    while task.active():
        time.sleep(30)

    # Get the task status
    # Source: ChatGPT
    status = task.status()
    if status['state'] == 'COMPLETED':
        logging.info(f"{file_name} export completed successfully.")
    else:
        logging.info(f"Export for {file_name} failed with error: {status['error_message']}")

def get_S2_images(study_site: str, params: dict, out_dir: str) -> None:
    """
    Gets and exports S2 images for a given study site into Google Drive.
    :param study_site: a string of study site name.
    :param params: a dictionary of parameters for the study site, consisting of:
                            study site crs, study site date range (of one day),
                            bounding box coordinates (maxLat, maxLon, minLat, minLon).
    :param out_dir: a string to the output directory.
    :return: None
    """

    # Check if S2 was already generated
    path = os.path.join(out_dir, study_site, "S2", "*.tif")
    if len(glob.glob(path)) > 0:
        return

    logging.info(f"Getting S2 image for: {study_site}")

    # Create a polygon from combined coord
    roi_gpd = _get_bbox_poly(params['bbox_coords'])

    # Convert it to ee object
    roi_ee = geemap.geopandas_to_ee(roi_gpd)

    # Get S2 gee image collection
    S2_col = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
                .filterDate(params['date_start'], params['date_end']) \
                .filterBounds(roi_ee)

    # Apply clipping
    S2_col_clipped = S2_col.map(lambda image: image.clip(roi_ee))
    logging.info(f"S2 crs: {S2_col_clipped.first().select(0).projection().crs().getInfo()}")
    logging.info(f"PS crs: {params['site_crs']}")

    # Aggregate IDs as a list, get unique values
    names = S2_col_clipped.aggregate_array('system:id').getInfo()
    names_unique = list(set([name.split('/')[2].split("_")[0] for name in names]))

    # If images are tiles of the same image, mosaic and export them
    if len(names_unique) == 1:
        # Mosaic and clip the image collection
        S2_mosaic = S2_col_clipped.mosaic().clip(roi_ee).select("B.*")

        # Mosaics are on default WGS86, so reprojecting them to local UTM based on PS crs
        S2_mosaic_repr = S2_mosaic.reproject(crs=params['site_crs'], scale=10)

        # Export resulting mosaic
        export_img(S2_mosaic_repr, study_site, f'{names_unique[0]}_{study_site}', roi_ee)

    else:
        print("INSPECT WHATS GOING ON")
        # Export images
        S2_list = S2_col_clipped.toList(S2_col_clipped.size())

        for i in range(S2_list.size().getInfo()):
            # Get image name
            name = img.id().getInfo().split('/')[2].split("_")[0]
            print(name)

            if params['site_crs'] != img.select(0).projection().crs().getInfo():
                img = ee.Image(S2_list.get(i)).select("B.*").reproject(crs=params['site_crs'], scale=10)

            else:
                img = ee.Image(S2_list.get(i)).select("B.*")

            # export_img(img, db, studySite, f'{name}_{studySite}')

def _get_bbox_poly(coords: tuple, crs: str="EPSG:4326") -> gpd.GeoDataFrame:
    """
    From give coordinates (TopRightCornerLat, TopRightCornerLon, BottomLeftCornerLat, BottomLeftCornerLon) in creates bounding box polygon.
    :param coords: string of the mentioned coordinates
    :param crs: string of crs system
    :return: geopandas polygon of bounding box
    """
    maxLat, maxLon, minLat, minLon = coords

    # redefine coords
    bbox_coords = [
        (minLon, minLat),
        (maxLon, minLat),
        (maxLon, maxLat),
        (minLon, maxLat),
        (minLon, minLat)
    ]

    # Create polygon
    polygon = Polygon(bbox_coords)
    polygon_gpd = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon])

    return polygon_gpd
