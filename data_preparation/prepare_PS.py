import os
from xml.dom import minidom
import logging

import numpy as np
import rasterio

logging.basicConfig(level=logging.INFO)

"""
Functions to obtain metadata from PS and convert it to TOAReflectance out of radiance
"""

def _get_metadata_path(path: str) -> str:
    """
    Compiles path to metadata file.
    :param path: path to the PS tif file.
    :return: string with a path to metadata xml file.
    """
    if 'clip' in path:
        return os.path.splitext(path)[0][:-5] + '_metadata_clip.xml'
    return os.path.splitext(path)[0] + '_metadata.xml'

def _get_conversion_coeffs(xml_path: str) -> list[float]:
    """
    Parses metadata XML file to get Top of Atmosphere (TOA) radiance-to-reflectance
    coefficient per band.
    Source: https://notebook.community/planetlabs/notebooks/jupyter-notebooks/in-class-exercises/convert-radiance-to-reflectance/convert-radiance-to-reflectance
    :param xml_path: a path to metadata XML file.
    :return: a list with conversion coefficients for each band in order of BGRN.
    """
    try:
        xmldoc = minidom.parse(xml_path)
        nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")

        # XML parser refers to bands by numbers 1-4
        coeffs = []
        for node in nodes:
            bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
            if bn in ['1', '2', '3', '4']:
                value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
                coeffs.append(float(value))
        return coeffs
    except Exception as e:
        raise ValueError(f"Error occurred while parsing XML metadata file: {e}")

def _convert_to_reflectance(coeffs: list[float], data: np.ndarray, scale_factor: int = 10000) -> np.ndarray:
    """
    Converts TOA radiance to reflectance with a given scaling coefficients.
    Source: https://notebook.community/planetlabs/notebooks/jupyter-notebooks/in-class-exercises/convert-radiance-to-reflectance/convert-radiance-to-reflectance
    :param coeffs: a list of scaling coefficients.
    :param data: a numpy array with stacked band data.
    :param scale_factor:  a scale_factor to save memory.
    :return: a numpy array with reflectance values
    """

    # Copy original data array dimensions
    data_converted = np.zeros_like(data)

    # Convert to TOA reflectance
    for i in range(data.shape[0]):
        data_converted[i] = data[i] * coeffs[i] * scale_factor

    return data_converted

def get_PS_TOARef(path: str) -> None:
    """
    Updates provided tif file with Top Of Atmosphere (TOA) Reflectacne instead of radiance.
    :param path: a string path to the tif file
    :return: None
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"The file '{path}' does not exist.")

    # Read in raster
    with rasterio.open(path) as src:
        # Get metadata
        meta = src.meta

        # Extract band names
        band_descriptions = [src.descriptions[i] for i in range(meta['count'])]

        # Read tif bands
        data = src.read()

        # Get reflectance band coefficients from metadata
        xml_path = _get_metadata_path(path)
        coeffs = _get_conversion_coeffs(xml_path)

        # Convert to reflectance values
        data = _convert_to_reflectance(coeffs, data)

    with rasterio.open(path, 'w', **meta) as dest:
        for i in range(data.shape[0]):
            dest.write(data[i], i + 1)
            dest.set_band_description(i+1, band_descriptions[i])

    logging.info(f'{path} was converted to TOAReflectance.')