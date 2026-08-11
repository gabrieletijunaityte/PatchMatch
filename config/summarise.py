import glob
import logging
import os
from natsort import natsorted, natsort_keygen

import pandas as pd

from config.configs import Configs

"""
This script extracts information from all configuration files and summarises them into csv table.
"""


def update_configurations(path='config/summarised_configs.csv', config_dir='config/'):
    # Get all configuration files
    config_paths = natsorted(glob.glob(config_dir + '/*.json'))

    # Check if configuration summary file exists
    if os.path.exists(path):
        configurations_df = pd.read_csv(path)
    else:
        logging.info('Creating new configurations file')
        configurations_df = pd.DataFrame()

    saved_configs = [] if len(configurations_df) == 0 else configurations_df.run_name.to_list()

    # List to save missing info
    rows = []

    # Iterate through all configuration files
    for config_path in config_paths:
        # Read configs
        print(f'Summarising configuration from {config_path}')
        config = Configs(config_path, create=False)

        loss_param = (f'margin: {config.loss_margin}' if config.loss_margin is not None else f'temperature: {config.loss_temp}' if config.loss_temp else None)

        normalise = ('Bandwise-normalised' if config.normalise and config.bandwise else 'Normalised' if config.normalise else 'Scaled')

        pre_trained = config.get_pre_trained_summary()

        # If config is not saved yet, save it
        if config.run_name not in saved_configs:
            row = {
                'PS_encoder': config.PS_encoder,
                'S2_encoder': config.S2_encoder,
                'shared_backbone': config.backbone,
                'projection_head': f'{config.projection_head} ({4 * config.hidden_dim} hid_dim)' if config.projection_head in ["linear", "simclr"] else None,
                'similarity': config.sim_name,
                'loss': config.loss,
                'loss_params': loss_param,
                'run_name': config.run_name,
                'pretrained': pre_trained,
                'lr': config.lr,
                'note': config.note,
                'normalise': normalise,
                'augmentation_strat': config.augmentations,
                'train_batch_size': config.batch_size,
                'allocated train epochs': config.epochs,
            }
            rows.append(row)
            saved_configs.append(config.run_name)

    if len(rows) != 0:
        # Convert into dataframe
        df = pd.DataFrame(rows)

        # Format keywords
        df = replacements(df)

        # Combine and save
        df = pd.concat([configurations_df, df], ignore_index=True)

        df = df.sort_values(by="run_name", key=natsort_keygen())

        df.to_csv(path, index=False)
        logging.info('Configuration file updated.')


def replacements(df):
    """Replacements of naming conventions"""
    replacements_dict = {
        'PSResNet18': 'ResNet18',
        'PSResNet50': 'ResNet50',
        'S2ResNet18': 'ResNet18',
        'S2ResNet50': 'ResNet50',
        'ResNet18': 'Resnet18',
        'Cosine_similarity': 'Cosine',
        'Cosine_Embedding_loss': 'Cosine Similarity Loss',
        'BCE_lin_loss': 'BCE Loss for Linear Learnable Difference Similarity',
        'Clip_Loss': 'Clip Loss',
        'PSPreEncoderSingle': 'Conv-Layer + AvgPool-layer',
        'S2PreEncoderSingle': 'Conv-Layer',
        'PSPreEncoder3Layer': '3-layer CNN',
        'S2PreEncoder3Layer': '3-layer CNN'
    }

    df = df.replace(replacements_dict).infer_objects(copy=False)

    df = df.sort_values(by=['loss', 'run_name'])
    return df


if __name__ == '__main__':
    update_configurations(path='summarised_configs.csv', config_dir='.')
