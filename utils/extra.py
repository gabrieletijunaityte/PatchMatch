import gc
import glob
import time

from osgeo import gdal
import numpy as np
import torch
from rasterio import open as ropen
from torchvision.transforms import v2

from data_preparation.data_inspection import load_min_max


def get_PS_paths(dir: str) -> list[str]:
    """
    Gets paths of all PS scenes in the given directory.
    :param dir: a string with a directory path.
    :return: a list
    """
    # Get all PS files
    return [i for i in glob.glob(f'{dir}/*/*.tif') if "udm" not in i]


def inspect_query(query: str, params: tuple) -> str:
    """
    Formats a query with parameters for inspection.
    """
    formatted_query = query
    for param in params:
        # Handle string parameters with escaping
        if isinstance(param, str):
            param = "'" + param.replace("'", "''") + "'"
        # Handle NULL values
        elif param is None:
            param = "NULL"
        # Other types are converted to strings directly
        else:
            param = str(param)
        # Replace the first '?' with the parameter
        formatted_query = formatted_query.replace("?", param, 1)
    return formatted_query


def open_file_as_tensor(path):
    img = ropen(path).read().astype(np.int16)
    tensor = v2.ToImage()(img).permute(1, 2, 0)
    return tensor


def open_file_as_tensor_norm(path, norm_mode='S2', min_max=None):
    tensor = open_file_as_tensor(path)
    if norm_mode in ['S2', 'PS']:
        mins, maxs = load_min_max(f'output/data_inspection/{norm_mode}_stats_01_99.csv', bandwise=True)
        tensor = normalisation(tensor, mins, maxs)
    elif norm_mode == 'self_PS_rgb':
        tensor = (tensor - tensor[:3, :, :].min()) / (tensor[:3, :, :].max() - tensor[:3, :, :].min())
    elif norm_mode == 'self_S2_rgb':
        tensor = (tensor - tensor[1:4, :, :].min()) / (tensor[1:4, :, :].max() - tensor[1:4, :, :].min())
    elif len(norm_mode) == 2:
        tensor = (tensor - norm_mode[0]) / (norm_mode[1] - norm_mode[0])
    else:
        raise ValueError(f'Unsupported normalisation mode: {norm_mode}')

    return tensor


def normalisation(tensor_array, min_vals, max_vals):
    tensor_array = np.clip(tensor_array, min_vals, max_vals)
    tensor_array = (tensor_array - min_vals) / (max_vals - min_vals)

    return tensor_array


def cleanup():
    # source: https://www.analyticsvidhya.com/blog/2024/10/memory-efficient-model-weight-loading-in-pytorch/
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        time.sleep(3)
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        time.sleep(3)


def compare_model_weights(model1, model2):
    if isinstance(model1, dict) and isinstance(model2, dict):
        state_dict1 = model1
        state_dict2 = model2
    else:
        state_dict1 = model1.state_dict()
        state_dict2 = model2.state_dict()

    missmatch_counts = 0
    for key in state_dict1.keys():
        if key in state_dict2:
            if not torch.equal(state_dict1[key], state_dict2[key]):
                print(f"Mismatch in layer: {key}")
                missmatch_counts = missmatch_counts + 1
        else:
            print(f"Key {key} not found in second model")
            missmatch_counts = missmatch_counts + 1
    if missmatch_counts == 0:
        return True
    else:
        return False


def build_pyramids(image_path: str) -> None:
    """
    Build gdal overviews (pyramids) for a given raster path.
    Sources:
    https://gdal.org/en/latest/tutorials/raster_api_tut.html
    https://stackoverflow.com/questions/60954617/how-to-build-internal-overviews-with-python-gdal-buildoverviews
    https://stackoverflow.com/questions/68025043/adding-a-progress-bar-to-gdal-translate

    :param image_path: an image path.
    :return: none
    """
    gdal.DontUseExceptions()

    # Open the tif file
    image = gdal.Open(image_path, gdal.GA_ReadOnly)

    # Set  configurations
    gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')

    # Build pyramids with nearest resampling for given scales
    # With progress bar
    image.BuildOverviews('NEAREST', [4, 16, 64, 128], gdal.TermProgress_nocb)

    # Close the image
    del image
