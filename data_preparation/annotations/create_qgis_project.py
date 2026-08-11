import sys
import os

import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from database import ProjectDB
from data_preparation.annotations.annotation_data import dummy_annotation_file
from config.configs import Configs

logging.basicConfig(level=logging.INFO)

def add_RasterLayer(file_path: str, project, name: str, style_dir: str='data/styles') -> None:
    """
    Adds raster to QGIS project based on the layer name
    :param file_path: a path to raster (tif file).
    :param project: a qgis project object.
    :param name:
    :return: None
    Parameters
    ----------
    file_path
    project
    name
    style_dir

    Returns
    -------

    """
    # Select style name based on the layer name
    style_path = os.path.join(style_dir, f'{name}.qml')

    # Create layer and add it to the project
    raster_layer = QgsRasterLayer(file_path, name)
    project.addMapLayer(raster_layer)

    # Read style variables
    a, b, c = read_style(style_path)

    # Apply the style from style file
    raster_layer.readStyle(a, b, c, QgsMapLayer.AllStyleCategories)

def read_style(style_path: str) -> tuple:
    """
    Reads style parameters from a style path
    Source: inspired by ChatGPT with prompt:
            using qgis core for python how can I apply style file .qml to a QgsRasterLayer.
    :param style_path:
    :return: a tuple of:
        documentElement, empty error line, context
    """
    doc = QDomDocument()
    with open(style_path, 'r') as style_file:
        style_content = style_file.read()
        doc.setContent(style_content)

    context = QgsReadWriteContext()

    return doc.documentElement(), '-', context

def qgis_to_annotate(PS_img: str, db: ProjectDB, config: Configs) -> None:
    """
    Create a project with PS, S2 images and corresponding vector file for easy manual annotating process.
    :param PS_img: a PS image name
    :param db: a database object
    :param config: a configuration dictionary, containing directory paths.
    :return: None
    """
    # Get directory paths
    styles_dir = config.styles_dir
    project_dir = config.projects_dir
    ndi_dir = config.ndi_dir
    annotation_dir = config.annotations_dir

    # Check if project already exists and skip if it does
    studySite = db.get_study_site(PS_img)
    project_path = os.path.join(project_dir, studySite, f"{PS_img[:-4]}.qgz")

    if os.path.isfile(project_path):
        return

    # Open application without gui
    app = QgsApplication([], GUIenabled=False)
    app.initQgis()

    # Create project
    project = QgsProject.instance()
    # Get target StudySite crs
    site_crs = db.get_site_crs(studySite)
    target_crs = QgsCoordinateReferenceSystem(site_crs)
    # Reproejct project to target crs
    project.setCrs(QgsCoordinateReferenceSystem(target_crs))

    # Get full paths to rasters
    PS_img_path = db.get_path(PS_img)
    # Base S2 image on PS scene
    S2_img = db.get_S2_from_PS(PS_img)
    S2_img_path = db.get_path(S2_img)

    # Add Scene RGB rasters
    add_RasterLayer(file_path=S2_img_path, project=project,name="S2", style_dir=styles_dir)
    add_RasterLayer(file_path=PS_img_path, project=project,name="PS", style_dir=styles_dir)

    # Add index layers
    PS_ndi_path = os.path.join(ndi_dir, studySite, "ndi_" + PS_img)
    S2_ndi_path =  os.path.join(ndi_dir, studySite, "ndi_" + S2_img)
    add_RasterLayer(file_path=S2_ndi_path, project=project, name="S2_NDI", style_dir=styles_dir)
    add_RasterLayer(file_path=PS_ndi_path, project=project,name="PS_NDI", style_dir=styles_dir)

    # Define annotation path
    geojson_name = os.path.splitext(PS_img)[0] + ".geojson"
    geojson_dir = os.path.join(annotation_dir, studySite)
    geojson_path = os.path.join(geojson_dir, geojson_name)
    os.makedirs(geojson_dir, exist_ok=True)

    # If annotation file exist, skip the rest
    if not os.path.isfile(geojson_path):
        dummy_annotation_file(geojson_path, site_crs=site_crs)

    # Create vector layer
    vector_layer = QgsVectorLayer(geojson_path, "Annotations")

    # Set crs to target crs
    vector_layer.setCrs(target_crs)

    # Add vector layer to the project
    project.addMapLayer(vector_layer)

    # get style variables
    a, b, c = read_style(os.path.join(styles_dir, 'Arrows.qml'))

    # Set the style
    vector_layer.readStyle(a, b, c, QgsMapLayer.AllStyleCategories)

    # Create studySite annotation folder if it does not exist
    os.makedirs(os.path.join(project_dir, studySite), exist_ok=True)

    # Save the project
    project.write(project_path)
    logging.info(f"Project saved to {project_path}")

    # Close off the project
    project.removeAllMapLayers()
    app.exitQgis()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: create_qgis_project.py <PS_img>")
        sys.exit(1)

    PS_path = sys.argv[1]
    config_path = sys.argv[2]

    config = Configs(config_path)

    PS_dir = config.PS_dir
    S2_dir = config.S2_dir

    from qgis.core import QgsApplication, QgsProject
    from qgis.core import QgsVectorLayer, QgsCoordinateReferenceSystem
    from qgis.core import QgsRasterLayer, QgsReadWriteContext, QgsMapLayer
    from PyQt5.QtXml import QDomDocument

    # Open database
    db = ProjectDB.ProjectDB(config.DB_path, config.PS_dir, config.S2_dir, config.test_poly_dir, config.validate_poly_dir)

    # Call the qgis_to_annotate function
    qgis_to_annotate(PS_path, db, config)



