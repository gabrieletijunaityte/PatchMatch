import os
import logging

import rasterio
import numpy as np

logging.basicConfig(level=logging.INFO)

def _calculate_ndi(band1_array: np.ndarray, band2_array: np.ndarray, scale_factor: int = 1000, nan_value: int = -9999) -> np.ndarray:
    """
    Calculates Normalised Difference Index (NDI) for given bands with a formula:
        (band2 - band1) / (band2 + band1) * scale_factor
        Arrays must have same dimensions.

    Parameters
    ----------
    band1_array - a numpy array with values of band1.
    band2_array - a numpy array with values of band2.
    scale_factor - a scale_factor to save memory.
    nan_value - a no data value.

    Returns
    -------
    a numpy array with calculated NDI values.
    """

    # Create same size empty array with -9999
    ndi = np.ones_like(band1_array) * nan_value

    # Find where ndi is valid (not a 0 in either bands or sum is not a 0
    # (to avoid division by 0))
    valid_indices = (((band1_array + band2_array) != 0)
                     & (band1_array != 0)
                     & (band2_array != 0))


    # Calculate NDI
    raw_ndi = ((band2_array[valid_indices] - band1_array[valid_indices]) / (
                band1_array[valid_indices] + band2_array[valid_indices]))

    ndi[valid_indices] = np.round(raw_ndi * scale_factor)

    return ndi

def save_ndi_raster(path: str, out_path: str, platform: str, dtype: str = 'int16', nan_value : int = -9999) -> None:
    """
    A function to calculate Normalised Difference Index for given raster (path).
    Source: https://zectre.github.io/geospatialpython/2021/08/08/NDVI-With-Rasterio.html

    Parameters
    ----------
    path - a path to a tif raster.
    out_path - a path to save the original raster with a new band
    platform - a string specifying platform. Possible values: "S2" or "PS".
    dtype - desired raster output dtype.
    nan_value - no data value.

    Returns
    -------
    None
    """

    # Define platform-to-band mappings
    platform_mapping = {
        "PS": (0, 3),
        "S2": (1, 7)
    }

    if not os.path.exists(path):
        raise FileNotFoundError(f"The file '{path}' does not exist.")
    if platform not in platform_mapping:
        raise ValueError(f"Unsupported platform: '{platform}'.")

    # Get bands if platform is specified
    band1_index, band2_index = platform_mapping[platform]

    # Read in raster
    with rasterio.open(path) as src:
        # Get metadata
        meta = src.meta

        # Change dtype and no data value
        meta.update(dtype=dtype, nodata=nan_value, count=1)

        # Read tif bands
        data = src.read().astype(dtype)

        # Calculate NDI
        band1 = data[band1_index]
        band2 = data[band2_index]
        ndi = _calculate_ndi(band1, band2, nan_value=nan_value).astype(dtype)

        # Create/check for the output directory
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Create a new file with the calculated ndi
        with rasterio.open(out_path, 'w', **meta) as dest:
            dest.write(ndi, 1)
            dest.set_band_description(1, 'ndi')

    logging.info(f"New file with ndi saved to: {out_path}")