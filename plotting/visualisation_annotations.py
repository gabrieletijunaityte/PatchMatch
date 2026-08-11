import matplotlib.pyplot as plt
import rasterio
import warnings
from matplotlib.patches import ConnectionPatch
import numpy as np
import geopandas as gpd
import pandas as pd
from rasterio.features import shapes
from data_preparation.annotations.annotation_data import parse_endpoint_coords
from database.ProjectDB import ProjectDB
from config.configs import Configs

plt.style.use('seaborn-v0_8')

"""Script to create annotation example figure in the report"""


# Normalize function for better contrast
def normalize(img, upper=1800):
    # Clip and normalize the image to the range [0, 1]
    clipped = np.clip((img - img.min()) / (upper - img.min()), 0, 1)

    # Separate the BGR channels
    b = clipped[:, :, 0]
    g = clipped[:, :, 1]
    r = clipped[:, :, 2]

    # Reorder the channels to RGB (restack)
    return np.stack([r, g, b], axis=-1)


# Mask out black/no-data areas
def mask_out(img, threshold=0.01):
    mask = np.all(img < 0.001, axis=-1)
    img[mask] = [1, 1, 1]
    return img


if __name__ == "__main__":
    config = Configs('config/proposed.json')

    db = ProjectDB(config.DB_path, config.PS_dir, config.S2_dir, config.test_poly_dir, config.validate_poly_dir)

    # Open geojson path
    annotations = gpd.read_file('data/annotations/Accra_2/20181031_095847_0f43_3B_AnalyticMS_clip.geojson')

    # Extract PS and S2 (lat, lon) coordinates from gpd dataframe
    coords = parse_endpoint_coords(annotations)
    coords = coords[1:]

    df = pd.DataFrame({"PS_Lon": [ps[1] for ps, s2 in coords], "PS_Lat": [ps[0] for ps, s2 in coords],
                       "S2_Lon": [s2[1] for ps, s2 in coords], "S2_Lat": [s2[0] for ps, s2 in coords], "Matched": 1, 'Color': '#6aade4'})

    # Paths to your TIFF images
    path_PS = 'data/PS/Accra_2/20181031_095847_0f43_3B_AnalyticMS_clip.tif'
    path_S2 = 'data/S2/Accra_2/20181031T101139_Accra_2.tif'

    # Set up figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 16))
    plt.subplots_adjust(hspace=0.05)

    frame = [181800, 184000, 629000, 630100]

    # Open and display PlanetScope image
    with rasterio.open(path_PS) as src_PS:
        bounds_PS = src_PS.bounds
        height_PS, width_PS = src_PS.height, src_PS.width
        transform_PS = src_PS.transform
        crs_PS = src_PS.crs
        data_PS = src_PS.read()[:-1]
        extent = [src_PS.bounds[0], src_PS.bounds[2], src_PS.bounds[1], src_PS.bounds[3]]

        band = src_PS.read(1, masked=True)

        # Create binary mask: 1 where valid, 0 where invalid
        if hasattr(band, 'mask'):
            mask = ~band.mask
        else:
            mask = np.ones(band.shape, dtype=bool)

        mask_int = mask.astype(np.uint8)

        results = ({'properties': {'raster_val': v}, 'geometry': s} for s, v in shapes(mask_int, transform=transform_PS) if v == 1)

        # Create GeoDataFrame from those features
        geoms = list(results)
        gdf_mask = gpd.GeoDataFrame.from_features(geoms, crs=crs_PS)

        valid_area = gdf_mask.union_all()

    gdf_PS = gpd.GeoDataFrame(df, crs=crs_PS, geometry=gpd.points_from_xy(df.PS_Lat, df.PS_Lon))
    gdf_PS = gdf_PS[gdf_PS.geometry.within(valid_area)]

    gdf_S2 = gpd.GeoDataFrame(gdf_PS, crs=crs_PS, geometry=gpd.points_from_xy(gdf_PS.S2_Lat, gdf_PS.S2_Lon))

    # Process PS image
    data_PS = np.transpose(data_PS, (1, 2, 0))
    data_PS = normalize(data_PS)
    data_PS = mask_out(data_PS)

    # Display PS image
    ax1.imshow(data_PS, extent=extent)
    ax1.set_xlim(frame[:2])
    ax1.set_ylim(frame[2:])

    with rasterio.open(path_S2) as src_S2:
        if src_S2.crs != crs_PS:
            warnings.warn("Warning: Images have different coordinate systems. Cropping may not be accurate.")

        # Read S2 data within window
        data_S2 = src_S2.read()[1:4]
        bounds_S2 = src_S2.bounds
        transform_S2 = src_S2.transform
        crs_S2 = src_S2.crs
        extent = [src_S2.bounds[0], src_S2.bounds[2], src_S2.bounds[1], src_S2.bounds[3]]

    frame = [181800, 184000, 629400, 630500]

    # Process S2 image
    data_S2 = np.transpose(data_S2, (1, 2, 0))
    data_S2 = normalize(data_S2)
    data_S2 = mask_out(data_S2)

    ax2.imshow(data_S2, extent=extent)

    ax2.set_xlim(frame[:2])
    ax2.set_ylim(frame[2:])

    # Connect points with ConnectionPatch
    for idx, row in gdf_PS.iterrows():
        ps_x, ps_y = row.PS_Lat, row.PS_Lon
        s2_x, s2_y = row.S2_Lat, row.S2_Lon
        line_color = '#6aade4'
        con = ConnectionPatch(xyA=(ps_x, ps_y), xyB=(s2_x, s2_y), coordsA="data", coordsB="data", axesA=ax1, axesB=ax2, color=line_color, arrowstyle="->", alpha=1, linewidth=5, mutation_scale=20)
        fig.add_artist(con)

    plt.tight_layout(rect=[1, 1, 1, 1])
    plt.subplots_adjust(hspace=0.05)

    ax1.grid(alpha=0.2)
    ax2.grid(alpha=0.2)

    ax1.tick_params(axis='both', which='both', left=True, right=True, top=True, bottom=True, labelleft=True, labelright=False, labelbottom=False, labeltop=False, labelsize=24)
    ax2.tick_params(axis='both', which='both', left=True, right=True, top=True, bottom=True, labelleft=True, labelright=False, labelbottom=True, labeltop=False, labelsize=24)

    plt.savefig("../annotations.png")
    plt.show()
