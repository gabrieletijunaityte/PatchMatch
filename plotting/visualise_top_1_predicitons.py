import os
import warnings
import random
from testing.retrieval_metrics import save_local_top_1_pairs

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import shapes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter
from shapely.geometry import Point
from shapely.geometry import box

from config.configs import Configs

plt.style.use('seaborn-v0_8')
"""Scripts to create top-1 prediction figures"""


def normalize(img, upper=1800):
    """Normalize function for better contrast"""
    # Clip and normalize the image to the range [0, 1]
    clipped = np.clip((img - img.min()) / (upper - img.min()), 0, 1)

    # Separate the BGR channels
    b = clipped[:, :, 0]
    g = clipped[:, :, 1]
    r = clipped[:, :, 2]

    # Reorder the channels to RGB (restack)
    return np.stack([r, g, b], axis=-1)


def mask_out(img):
    mask = np.all(img < 0.001, axis=-1)
    img[mask] = [1, 1, 1]
    return img


def cooridnate_format(x, pos):
    """ Converts coordinates to thousands format"""
    return f'{int(x):}'


def visualise_top_1(site, df, frame=None, legend=True, name=None, return_figure=False, id_val=None, radius_plot=True):
    green = '#BFB304FF'
    red = '#E04B28FF'

    # Load data
    if site == 'Accra':
        path_PS = 'data/PS/Accra_2/20181031_095850_0f43_3B_AnalyticMS_clip.tif'
        path_S2 = 'data/S2/Accra_2/20181031T101139_Accra_2.tif'
        radius = 769 * 0.9
    else:
        path_PS = 'data/PS/Marmara/20210519_081347_67_2251_3B_AnalyticMS_clip.tif'
        path_S2 = 'data/S2/Marmara/20210519T084601_Marmara.tif'
        radius = 1934 * 0.4

    df['Color'] = 'str'
    df.loc[df['Matched'] == 1, 'Color'] = green
    df.loc[df['Matched'] == 0, 'Color'] = red

    # Set up figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    plt.subplots_adjust(hspace=0.05)

    with rasterio.open(path_PS) as src_PS:
        bounds_PS = src_PS.bounds
        transform_PS = src_PS.transform
        crs_PS = src_PS.crs
        data_PS = src_PS.read()[:-1]
        extent = [src_PS.bounds[0], src_PS.bounds[2], src_PS.bounds[1], src_PS.bounds[3]]
        frame = frame or extent

        band = src_PS.read(1, masked=True)

        # Create binary mask: 1 where valid, 0 where invalid
        if hasattr(band, 'mask'):
            mask = ~band.mask
        else:
            mask = np.ones(band.shape, dtype=bool)

        # Convert to 8-bit integer for shapes()
        mask_int = mask.astype(np.uint8)

        # Extract polygon features of valid areas
        results = ({'properties': {'raster_val': v}, 'geometry': s} for s, v in shapes(mask_int, transform=transform_PS)
                   if v == 1)

        # Create GeoDataFrame from those features
        geoms = list(results)

        minx, maxx, miny, maxy = frame
        valid_area = box(minx, miny, maxx, maxy)

    gdf_PS = gpd.GeoDataFrame(df, crs=crs_PS, geometry=gpd.points_from_xy(df.PS_Lat, df.PS_Lon))
    gdf_PS = gdf_PS[gdf_PS.geometry.within(valid_area)]

    gdf_S2 = gpd.GeoDataFrame(gdf_PS, crs=crs_PS, geometry=gpd.points_from_xy(gdf_PS.S2_Lat, gdf_PS.S2_Lon))
    gdf_S2_all = gpd.GeoDataFrame(df, crs=crs_PS, geometry=gpd.points_from_xy(df.S2_Lat, df.S2_Lon))
    gdf_S2_all = gdf_S2_all[gdf_S2_all.geometry.within(valid_area)]

    if id_val is None:
        id_val = random.randint(0, len(gdf_PS))

    if radius_plot:
        selected_point_row = gdf_PS.iloc[id_val, :]
        selected_point = selected_point_row.iloc[-1]

        selected_point_S2 = Point(selected_point_row.S2_Lat, selected_point_row.S2_Lon)
        search_radius = gpd.GeoDataFrame(geometry=[selected_point.buffer(radius)])
        selected_point = gpd.GeoDataFrame(geometry=[selected_point])
        selected_point_S2 = gpd.GeoDataFrame(geometry=[selected_point_S2])

    # Process PS image
    data_PS = np.transpose(data_PS, (1, 2, 0))
    data_PS = normalize(data_PS)
    data_PS = mask_out(data_PS)

    frame = frame or extent

    # Display PS image
    ax1.imshow(data_PS, extent=extent)
    if radius_plot:
        search_radius.plot(ax=ax1, facecolor='white', alpha=0.3)
    ax1.set_xlim(frame[:2])
    ax1.set_ylim(frame[2:])

    s2_full = True
    if s2_full:
        # Open and display Sentinel-2 image
        with rasterio.open(path_S2) as src_S2:
            if src_S2.crs != crs_PS:
                warnings.warn("Warning: Images have different coordinate systems. Cropping may not be accurate.")

            # Create window for S2 based on PS bounds
            window_S2 = from_bounds(*bounds_PS, src_S2.transform)
            window_S2 = window_S2.intersection(rasterio.windows.Window(0, 0, src_S2.width, src_S2.height))

            # Read S2 data within window
            data_S2 = src_S2.read(window=window_S2)[1:4]
            transform_S2 = src_S2.window_transform(window_S2)
            bounds_S2 = rasterio.windows.bounds(window_S2, src_S2.transform)
    else:
        with rasterio.open(path_S2) as src_S2:
            if src_S2.crs != crs_PS:
                warnings.warn("Warning: Images have different coordinate systems. Cropping may not be accurate.")

            # Read S2 data within window
            data_S2 = src_S2.read()[1:4]
            bounds_S2 = src_S2.bounds
            transform_S2 = src_S2.transform
            crs_S2 = src_S2.crs
            extent = [src_S2.bounds[0], src_S2.bounds[2], src_S2.bounds[1], src_S2.bounds[3]]

    # Process S2 image
    data_S2 = np.transpose(data_S2, (1, 2, 0))
    data_S2 = normalize(data_S2)
    data_S2 = mask_out(data_S2)

    ax2.imshow(data_S2, extent=extent)

    ax2.set_xlim(frame[:2])
    ax2.set_ylim(frame[2:])

    # Plot the points on each image
    gdf_PS.plot(ax=ax1, color=gdf_PS['Color'], markersize=10)
    gdf_S2_all.plot(ax=ax2, color=red, markersize=10)
    gdf_S2.plot(ax=ax2, color=gdf_PS['Color'], markersize=10)

    # Connect points with ConnectionPatch - this works better for connecting points across axes
    for idx, row in gdf_PS.iterrows():
        # Get coordinates
        ps_x, ps_y = row.PS_Lat, row.PS_Lon
        s2_x, s2_y = row.S2_Lat, row.S2_Lon

        line_color = green if row['Matched'] else red

        # Create a connection patch between the two points
        con = ConnectionPatch(xyA=(ps_x, ps_y), xyB=(s2_x,
                                                     s2_y), coordsA="data", coordsB="data", axesA=ax1, axesB=ax2, color=line_color, alpha=1, linewidth=3, zorder=1)

        # Add the connection patch to the figure
        fig.add_artist(con)

    # Plot MD example points
    if radius_plot:
        search_radius.plot(ax=ax2, facecolor='white', alpha=0.3, zorder=4)
        selected_point.plot(ax=ax1, facecolor='white', zorder=4)
        selected_point.plot(ax=ax2, facecolor='white', zorder=4)
        selected_point_S2.plot(ax=ax2, facecolor='#6aade4', zorder=4)

    ax1.set_xlabel('')
    ax1.set_xticklabels([])

    if legend:
        if radius_plot:
            legend_elements = [Line2D([0], [
                0], marker='o', linestyle='None', color='none', markeredgecolor='white', markerfacecolor='#6aade4', markersize=10, label='Match position'),
                               Line2D([0], [
                                   0], marker='o', linestyle='None', color='none', markerfacecolor='white', markersize=10, label='Query position'),
                               Line2D([0], [0], color=green, lw=3, label='Correctly matched MD patches'),
                               Line2D([0], [0], color=red, lw=3, label='Wrongly matched MD patched')]
        else:
            legend_elements = [Line2D([0], [0], color=green, lw=3, label='Correctly matched MD patches'),
                               Line2D([0], [0], color=red, lw=3, label='Wrongly matched MD patched')]
        ax2.legend(handles=legend_elements, loc=legend, fancybox=False, ncol=1, frameon=True, framealpha=0.3, labelcolor='white', prop=FontProperties(weight='bold', size=20))

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, hspace=0.05)

    ax1.yaxis.set_major_formatter(FuncFormatter(cooridnate_format))
    ax2.yaxis.set_major_formatter(FuncFormatter(cooridnate_format))

    ax1.grid(alpha=0.2)
    ax2.grid(alpha=0.2)

    ax1.tick_params(axis='both', which='both', left=True, right=True, top=True, bottom=True, labelleft=True, labelright=False, labelbottom=False, labeltop=False, labelsize=24)

    ax2.tick_params(axis='both', which='both', left=True, right=True, top=True, bottom=True, labelleft=True, labelright=False, labelbottom=True, labeltop=False, labelsize=24)

    if return_figure:
        return fig

    plt.savefig(f'../top_1_{name or site}.png')
    plt.show()


if __name__ == '__main__':
    config = Configs('config/proposed.json')
    df_dict = save_local_top_1_pairs(config, return_dic=True)

    # Local
    accra_local = visualise_top_1(site="Accra", df=df_dict['Accra_2'], frame=[160000, 174000, 616000, 622000], return_figure=False, id_val=3, legend=False, radius_plot=False)
    marmara_local = visualise_top_1("Marmara", df_dict['Marmara'], frame=[680000, 698000, 4512500, 4521000], legend=False, name="marmara_2", return_figure=False, id_val=15, radius_plot=False)

    # Drift-bound
    path = os.path.join(config.pred_dir, config.run_name, f'pair_top_1_predictions_radius_Accra_2_0.9.csv')
    df_dict = pd.read_csv(path)

    accra_drift = visualise_top_1("Accra", df_dict, frame=[160000, 174000, 616000, 622000], return_figure=False, radius_plot=True, id_val=15, legend="lower left")

    path = os.path.join(config.pred_dir, config.run_name, f'pair_top_1_predictions_radius_Marmara_0.4.csv')
    df_dict = pd.read_csv(path)

    marmara_drift = visualise_top_1("Marmara", df_dict, frame=[680000, 698000, 4512500, 4521000], legend='lower left', name="marmara_2", return_figure=False, id_val=15)
