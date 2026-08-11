from sklearn.metrics import top_k_accuracy_score
import numpy as np
import logging
import pandas as pd
import os
import glob
from natsort import natsort_keygen

from database import ProjectDB
from plotting.top_k_predictions import plot_top_k_predictions

logging.basicConfig(level=logging.INFO)

"""Functions that calculate global, local and drift bound retrieval metrics"""


def open_prediction_df(config):
    """Open prediction matrix if available, otherwise return True for missing prediction matrix."""
    # Check if predictions exists
    if not os.path.exists(config.pred_path):
        logging.info(f"Predictions are not available for '{config.run_name}' configuration'.")

        # True to obtain predictions in while loop
        return True

    # Read in predictions
    df_pred = pd.read_csv(config.pred_path, names=["PS", "S2", "Pred_sim", "Match_label"], dtype={"PS": str, "S2": str,
                                                                                                  "Pred_sim": float,
                                                                                                  "Match_label": int})
    return df_pred


def top_k_accuracy(similarities, k=3):
    """Calculates top-k accuracy fro given k and given similarity matrix."""
    similarity_array = np.array(similarities).reshape(1, -1)
    top_k_acc = top_k_accuracy_score(np.array([0]), similarity_array, k=k, labels=list(range(len(similarities))))
    return top_k_acc.item()


def get_position(df):
    """Gets the mean position of the true match given prediction matrix"""
    df = df.sort_values(by=['Pred_sim'], ascending=False, ignore_index=True)
    positions = df[df['Match_label'] == 1].index
    if len(positions) == 1:
        return positions[0] + 1
    elif len(positions) == 0:
        raise ValueError("Position could not be determined, because there was no true match")
    else:
        raise ValueError("Position could not be determined, because there were multiple true matches")


def avr_metrics(df, ks):
    """Obtains average retrieval metrics for given top-k's"""
    # Get PS patch IDs
    PS_IDs = df.PS.unique()

    # Initialise storage for metrics
    top_k_accuracies = {k: [] for k in ks}
    positions = []

    for PS_tile in PS_IDs:
        # Subset predictions for specific PS IDs and all possible S2 IDs
        subset = df[df.PS == PS_tile].sort_values(by="Match_label").sort_values(by="Match_label", ascending=False)

        # Convert to list for top-k function
        similarities = subset.Pred_sim.to_list()
        matches = subset.Match_label.to_list()

        # Check for errors in labels with none or multiple matches (should be only one S2 match for each PS)
        if matches[0] == 1 and sum(matches) == 1:
            for k in ks:
                top_k_accuracies[k].append(top_k_accuracy(similarities, k))
            positions.append(get_position(subset))
        else:
            raise ValueError(f"More than one match for ID = {PS_tile}")

    # Average results
        results = {}
    for k in ks:
        logging.info(f'Top-{str(k)} average accuracy:{np.mean(top_k_accuracies[k])}')
        results[f'{k}'] = np.mean(top_k_accuracies[k])

    mean_pos = np.mean(positions)
    logging.info(f'Mean position: {mean_pos}')
    results['mean_position'] = mean_pos

    return results


def calculate_global_top_k(config):
    """Calculates top-k's for given configuration setup at a global scale"""
    # Check for prediction output file
    df_pred = open_prediction_df(config)
    if df_pred is True:
        return True

    # Check if file existing
    top_k_path = os.path.join(config.results_dir, 'global_top_k_means.csv')
    if not os.path.exists(top_k_path):
        # Initialise if not
        initialise_top_k(df_pred, config.top_ks, config.run_name, top_k_path)
    else:
        # If it does, append metrics for config
        append_top_k(df_pred, config.top_ks, config.run_name, top_k_path)
    return False


def initialise_top_k(df_pred, top_ks, run_name, path, filters=None):
    """Initialises top-k result csv file"""
    if filters:
        df_pred = df_pred[df_pred["PS"].isin(filters[0]) & df_pred["S2"].isin(filters[1])]

    k_means = avr_metrics(df_pred, top_ks)
    k_means_df = pd.DataFrame.from_dict([k_means])
    k_means_df['run_name'] = run_name

    k_means_df.to_csv(path, header=True, index=False)
    logging.info(f'Top-K-Accuracy saved to {path}')


def append_top_k(df_pred, top_ks, run_name, path, filters=None):
    """Appends results to a given top-k results file"""
    # Read in metrics
    df = pd.read_csv(path)

    # Check if metrics saved for setup
    if run_name in df.run_name.to_list():
        return

    # Apply site filters if given
    if filters:
        df_pred = df_pred[df_pred["PS"].isin(filters[0]) & df_pred["S2"].isin(filters[1])]

    k_means = avr_metrics(df_pred, top_ks)
    k_means = pd.DataFrame.from_dict([k_means])
    k_means['run_name'] = run_name

    df = pd.concat([df, k_means])
    df = df.sort_values(by="run_name", key=natsort_keygen())
    df.to_csv(path, header=True, index=False)
    logging.info(f'Top-K-Accuracy saved to {path}')


def get_top_k_images(config, k, PS_patch_ID=None, plot=False, save=False):
    """Creates random or specified top-k prediction figures or returns top-k predictions"""
    # Open prediction path
    df = open_prediction_df(config)

    # Sample random PS ID if not provided
    if PS_patch_ID:
        # Check if images saved
        path = os.path.join(config.pred_fig_dir, f'top_{k}_pred_PS_{PS_patch_ID}.png')
        if save and os.path.exists(path):
            return
    else:
        PS_patch_ID = df.sample(1).PS.item()
        path = os.path.join(config.pred_fig_dir, f'top_{k}_pred_PS_{PS_patch_ID}.png')
        while os.path.exists(path):
            PS_patch_ID = df.sample(1).PS.item()
            path = os.path.join(config.pred_fig_dir, f'top_{k}_pred_PS_{PS_patch_ID}.png')

    # Subset to PS ID and all possible S2 IDs
    PS_df = df[df.PS == PS_patch_ID].copy()

    # Find the match row
    PS_match_row = PS_df[PS_df.Match_label == 1]

    # Sort all predictions on similarity
    PS_df.sort_values(by="Pred_sim", ascending=False, inplace=True)

    # Subset to top-k
    PS_df = PS_df[:k]
    # Put PS match row first and top-k predictions next
    df_reformated = pd.concat([PS_match_row, PS_df], ignore_index=True)

    # Make paths out of IDs
    df_reformated['PS'] = df_reformated['PS'].apply(lambda x: os.path.join(config.pt_dir, config.pt_template.replace('?', '{}').format(x, 'PS')))
    df_reformated['S2'] = df_reformated['S2'].apply(lambda x: os.path.join(config.pt_dir, config.pt_template.replace('?', '{}').format(x, 'S2')))

    if plot:
        plot_top_k_predictions(df_reformated, save=save, path=path, config=config)
    else:
        return df_reformated


def calculate_local_top_k(config):
    """For a given configuration, calculates local top-k accuracies per study site"""
    # Read in predictions
    df_pred = open_prediction_df(config)

    # Get test site names
    test_sites = [i.split("/")[-1].split(".")[0] for i in glob.glob(config.test_poly_dir + "/*")]

    # Open db to filter IDs based on study site
    db = ProjectDB.ProjectDB(config.DB_path, config.PS_dir, config.S2_dir, config.test_poly_dir, config.validate_poly_dir)

    # Obtain metrics per test site
    for site in test_sites:
        top_k_path = os.path.join(config.results_dir, f'local_{site}_top_k_means.csv')

        # Test site filters for IDs
        s2_IDs = [str(id[0]) for id in db.get_test_IDs_per_studysite(platform="S2", study_site=site)]
        ps_IDs = [str(id[0]) for id in db.get_test_IDs_per_studysite(platform="PS", study_site=site)]

        if not os.path.exists(top_k_path):
            initialise_top_k(df_pred, config.top_ks, config.run_name, top_k_path, filters=(ps_IDs, s2_IDs))
        else:
            append_top_k(df_pred, config.top_ks, config.run_name, top_k_path, filters=(ps_IDs, s2_IDs))


def save_local_top_1_pairs(config, return_dic=False):
    """Saves top-1 predictions at a local scope for each study event."""
    df_pred = open_prediction_df(config)

    test_sites = [i.split("/")[-1].split(".")[0] for i in glob.glob(config.test_poly_dir + "/*")]

    db = ProjectDB.ProjectDB(config.DB_path, config.PS_dir, config.S2_dir, config.test_poly_dir, config.validate_poly_dir)

    df_dict = {}

    # Create dir for predictions
    if not os.path.exists(os.path.join(config.pred_dir, config.run_name)):
        os.mkdir(os.path.join(config.pred_dir, config.run_name))

    for site in test_sites:
        path = os.path.join(config.pred_dir, config.run_name, f'pair_top_1_predictions_{site}.csv')
        if os.path.exists(path):
            df_dict[site] = pd.read_csv(path)
            continue

        pairs = {}

        # Get IDs just for study site
        s2_IDs = [str(id[0]) for id in db.get_test_IDs_per_studysite(platform="S2", study_site=site)]
        ps_IDs = [str(id[0]) for id in db.get_test_IDs_per_studysite(platform="PS", study_site=site)]

        # Filter out the prediction dataset for study site
        df = df_pred[df_pred["PS"].isin(ps_IDs) & df_pred["S2"].isin(s2_IDs)]

        PS_tiles = df.PS.unique()

        for PS_tile in PS_tiles:
            subset = df[df.PS == PS_tile].sort_values(by="Match_label").sort_values(by="Match_label", ascending=False)

            matches = subset.Match_label.to_list()

            if matches[0] == 1 and sum(matches) == 1:
                subset.sort_values(by="Pred_sim", inplace=True, ascending=False)
                pairs[subset.iloc[0, 0]] = subset.iloc[0, 1]
            else:
                raise ValueError

        if return_dic:
            df_dict[site] = get_top_1_pairs_df(db, pairs, save_path=path)
        else:
            get_top_1_pairs_df(db, pairs, save_path=path)

    if return_dic:
        return df_dict


def get_top_1_pairs_df(db, top_1_pairs, save_path=None):
    PS_IDs = list(top_1_pairs.keys())
    S2_IDs = list(top_1_pairs.values())
    matches_info = db.get_match_status(PS_IDs=PS_IDs, S2_IDs=S2_IDs)
    df = pd.DataFrame(matches_info, columns=["PS_Lon", "PS_Lat", "S2_Lon", "S2_Lat", "Matched"])
    if save_path:
        df.to_csv(save_path, index=False)
        logging.info(f'Prediction pairs were saved to {save_path}')
    else:
        return df


def calculate_drift_top_k(config, speeds, site):
    """Calculates drift-bound top-k accuracies for given study event and drift speeds"""
    # Check if file exists
    path = os.path.join(config.results_dir, f'drift_{site}_top_k_means_radius.csv')
    if os.path.exists(path):
        df_initial = pd.read_csv(path)

        # Check if results saved
        if config.run_name in df_initial.run_name.to_list():
            radii = df_initial.radius.to_list()
            speeds = [speed for speed in speeds if speed not in radii]
            if len(speeds) == 0:
                return
    else:
        df_initial = None

    # Filter based on radius and get metrics
    for speed in speeds:
        results = radius_filter(config, speed, [1, 2, 3, 5, 10, 50], site)
        results['run_name'] = config.run_name
        results['radius'] = speed

        df = pd.DataFrame.from_dict([results])
        if df_initial is not None:
            df = pd.concat([df, df_initial])

        df = df.sort_values(by=["run_name", "radius"], key=natsort_keygen())

        df.to_csv(path, header=True, index=False)
        logging.info(f'Top k accuracy per radius filtered data saved to {path}')
        df_initial = df



def radius_filter(config, speed, ks, study_site):
    """Filter out candidates outside the drfit bound radius"""
    # Open db
    db = ProjectDB.ProjectDB(config.DB_path, config.PS_dir, config.S2_dir, config.test_poly_dir, config.validate_poly_dir)

    # Get df of time difference
    df = pd.DataFrame(db.get_time_difference(site=study_site), columns=['PS_ID', 'PS_lat', 'PS_lon', 'match_S2_ID', 'time_dif_sec'])

    # Convert time to distance
    df['dist_meters'] = df.apply(lambda row: row['time_dif_sec'] * speed, axis=1)

    empty = []
    top_1_pair_map = {}

    # Open all predictions
    df_pred = open_prediction_df(config)

    top_k_accuracies = {k: [] for k in ks}
    positions = []

    for _, PS_ID, PS_lat, PS_lon, S2_ID, _, dist in df.itertuples():
        S2_IDs = db.get_points_in_radius(PS_lat, PS_lon, dist, study_site, plat="S2", mode="test")
        S2_IDs = [str(i[0]) for i in S2_IDs]

        subset = df_pred[(df_pred['PS'] == str(PS_ID)) & (df_pred['S2'].isin(S2_IDs))]

        if len(subset) == 0:
            top_1_pair_map[PS_ID] = None
            empty += [PS_ID]
            for k in ks:
                top_k_accuracies[k].append(None)
        else:
            subset = subset.sort_values("Pred_sim", ascending=False, ignore_index=True)
            top_1_pair_map[PS_ID] = subset.iloc[0, 1]

            if sum(subset.Match_label) == 0:
                for k in ks:
                    if k <= len(subset):
                        top_k_accuracies[k].append(0)
                    else:
                        top_k_accuracies[k].append(None)
            elif sum(subset.Match_label) == 1:
                positions.append(get_position(subset))
                for k in ks:
                    if k <= len(subset):
                        top_k_accuracies[k].append(1.0 if 1 in subset[:k].Match_label.to_list() else 0.0)
                    else:
                        top_k_accuracies[k].append(1.0 if subset.Match_label.to_list() else None)
            else:
                raise ValueError('Match sum should not be more than 1')

    # Average results
    results = {}
    ks_skipped = {k: 0 for k in ks}
    ks_skipped['empty'] = len(empty)
    for k in ks:
        ks_skipped[k] = sum(1 for i in top_k_accuracies[k] if i is None)
        top_k_accuracies[k] = [1 if i == 1 else 0 for i in top_k_accuracies[k]]
        logging.info(f'Top-{str(k)} average accuracy:{np.mean(top_k_accuracies[k])}')
        results[f'{k}'] = np.mean(top_k_accuracies[k])

    mean_pos = np.mean(positions)
    logging.info(f'Mean position: {mean_pos}')
    results['mean_position'] = mean_pos

    for key, value in ks_skipped.items():
        results[f'{key}_skipped'] = value

    # save top-1 pair maps
    path =  os.path.join(config.pred_dir, config.run_name, f'pair_top_1_predictions_radius_{study_site}_{speed}.csv')
    get_top_1_pairs_df(db, top_1_pair_map, save_path=path)

    return results
