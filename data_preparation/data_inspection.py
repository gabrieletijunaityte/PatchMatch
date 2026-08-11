import csv
import logging
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import rasterio
import rasterio as rio
import torch
from matplotlib import pyplot as plt
from natsort import natsorted, index_natsorted

"""
Functions to check no data values in the tiles.
"""

def save_no_data_inspection(paths, mode='train', dir='output/data_inspection/'):
    """Saves statistics of tiles with no data"""
    out_filename = os.path.join(dir, f'nodata_{mode}_tiles.csv')

    if os.path.exists(out_filename):
        return

    os.makedirs(dir, exist_ok=True)

    # Read paths that are in  {'train': {'tile_name' : ['PS_path', 'S2_path'] ... }} format
    paths = [path for tuple_paths in paths[mode].values() for path in tuple_paths]
    logging.info(f'For the {mode} set:')

    # Remove duplicates
    paths = list(dict.fromkeys(paths))

    # get no data tile names and save them
    get_nodata_tile_names(paths, out_filename)


def get_nodata_tile_names(paths, out_filename):
    """Gets the names of tiles containing no data."""
    non_full_tiles = []
    for path in paths:
        if check_for_nodata(path):
            non_full_tiles.append(path)

    with open(out_filename, 'w', encoding='UTF8', newline='') as f:
        wr = csv.writer(f)
        for path in non_full_tiles:
            wr.writerow([path])

    logging.info(f'there were {len(non_full_tiles)} tiles with nodata. Names have been written to {out_filename}\n')


def check_for_nodata(path):
    """Checks if tile (raster) has no data values"""
    with rasterio.open(path) as src:
        if src.nodata is None:
            nodata = 0
        else:
            nodata = src.nodata

        nans_count = np.sum((src.read() == src.nodata))
        if nans_count > 0:
            return True
        else:
            return False


def save_pixel_counts(paths, mode='train', dir='output/data_inspection/'):
    """Saves pixel counts per mode into csv"""
    # Parse paths that are list of lists like: [PS_path, S2_path]
    paths_PS = [path[0] for path in paths[mode].values()]
    paths_S2 = [path[1] for path in paths[mode].values()]

    # Get frequency tables of pixel values
    get_pixel_counts(paths_PS, mode, 'PS', dir)
    get_pixel_counts(paths_S2, mode, 'S2', dir)


def get_pixel_counts(tile_paths, mode, platform, dir='output/data_inspection/'):
    """Counts pixel values across tiles"""
    # https://gis.stackexchange.com/questions/382289/count-no-data-pixel-with-rioxarray
    # https://stackoverflow.com/questions/11290092/python-elegantly-merge-dictionaries-with-sum-of-values

    bands = 4 if platform == 'PS' else 13
    path_all = os.path.join(dir, f'all_{platform}_{mode}_px_counts.csv')

    for i in range(1, bands + 1):
        path = os.path.join(dir, f'band_{i}_{platform}_{mode}_px_counts.csv')
        if not os.path.exists(path):
            break
        elif i == bands:
            if not os.path.exists(path_all):
                break
            else:
                return

    os.makedirs(dir, exist_ok=True)

    counts = {f'band_{i}': Counter() for i in range(1, bands + 1)}
    counts['all'] = Counter()
    for tile_path in tile_paths:
        with rio.open(tile_path) as src:
            no_data = src.nodata

            for i in range(1, bands + 1):
                img = src.read(i)
                unique, count = np.unique(img, return_counts=True)
                temp_dict = dict(zip(unique, count))
                if no_data in unique:
                    temp_dict.pop(no_data)
                counts[f'band_{i}'] += Counter(temp_dict)
                counts['all'] += Counter(temp_dict)

    df = pd.DataFrame.from_dict(counts['all'], orient='index')
    df = df.reset_index()
    df.columns = ['value', 'counts']
    df.sort_values(by='value', inplace=True)
    df['cum_sum'] = df['counts'].cumsum()
    df.to_csv(path_all, index=False)

    for i in range(1, bands + 1):
        df = pd.DataFrame.from_dict(counts[f'band_{i}'], orient='index')
        df = df.reset_index()
        df.columns = ['value', 'counts']
        df.sort_values(by='value', inplace=True)
        df['cum_sum'] = df['counts'].cumsum()
        path = os.path.join(dir, f'band_{i}_{platform}_{mode}_px_counts.csv')
        df.to_csv(path, index=False)

        plt.bar(list(counts[f'band_{i}'].keys()), counts[f'band_{i}'].values(), color='g')
        plt.title(f'Band {i} pixel values distribution for {platform} {mode} tiles')
        path = os.path.join(dir, f'band_{i}_{platform}_{mode}_px_counts.png')
        plt.savefig(path)
        plt.close()

    logging.info(f'The counts of pixel values from {platform} {mode} tiles were saved {dir}')
    logging.info(f'The histogram of pixel values from {platform} {mode} tiles were saved {dir}')


def save_band_stats(paths, platform, p1=0.025, p2=0.975, dir='output/data_inspection'):
    """Saves pixel counts per band across tiles with given cut-off p values"""
    paths = natsorted(paths)

    out_name = os.path.join(dir, f'{platform}_stats_{str(p1).split('.')[-1]}_{str(p2).split('.')[-1]}.csv')
    if os.path.exists(out_name):
        return

    stats = []

    for i, path in enumerate(paths):
        counts = np.genfromtxt(path, delimiter=',', skip_header=1, usecols=(0, 1, 2))

        if p1 != 0 and p2 != 1:
            p1_val, p2_val = filter_percentiles(counts, p1, p2)
            filtered_counts = counts[(np.logical_and(counts[:, 0] >= p1_val, counts[:, 0] <= p2_val)), :-1]
        else:
            filtered_counts = counts

        mean, std = get_mean_std(filtered_counts)
        min_val, max_val = get_min_max(filtered_counts)

        band_pattern = re.search(r'band_(\d+)', path)
        if band_pattern:
            band = int(band_pattern.group(1))
        else:
            band = "all"

        stats.append({
            'Band': band,
            'Mean': mean,
            'Std': std,
            'Min': min_val,
            'Max': max_val})

    df = pd.DataFrame(stats)

    df.to_csv(out_name, index=False)

    logging.info(f'Statistics were saved to {out_name}')


def filter_percentiles(array, p1, p2):
    """Obtains actual pixel values that correspond to p1 and p2 percentiles"""
    summed = array[:, 1].sum()

    p1 = p1 * summed
    p2 = p2 * summed

    p1 = array[array[:, 2] >= p1][0, 0]
    p2 = array[array[:, 2] <= p2][-1, 0]

    return p1, p2


def get_mean_std(counts):
    """Gets statistics per counts"""
    mean = np.sum(counts[:, 0] * counts[:, 1]) / np.sum(counts[:, 1])
    sq_dif_from_mean = (counts[:, 0] - mean) ** 2
    std = np.sqrt(np.sum(sq_dif_from_mean * counts[:, 1]) / np.sum(counts[:, 1]))
    return mean, std


def get_min_max(counts):
    """Gets statistics per counts"""
    min_val = counts[:, 0].min()
    max_val = counts[:, 0].max()
    return min_val, max_val


def load_min_max(path, bandwise=True):
    """Loads in the calculated statistics"""
    df = pd.read_csv(path)
    df = df.sort_values(by='Band', key=lambda x: np.argsort(index_natsorted(df["Band"])))
    df = df.reset_index(drop=True)

    if bandwise:
        min = torch.tensor(df.loc[df['Band'] != 'all', 'Min']).view(-1, 1, 1)
        max = torch.tensor(df.loc[df['Band'] != 'all', 'Max']).view(-1, 1, 1)

    else:
        min = df.loc[df['Band'] == 'all', 'Min'].values[0]
        max = df.loc[df['Band'] == 'all', 'Max'].values[0]

    return min, max
