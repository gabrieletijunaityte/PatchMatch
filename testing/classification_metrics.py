import logging
import os

import lightning.pytorch as pl
import pandas as pd
import torch
from natsort import natsort_keygen

from utils.setups import setup_model, setup_dataloader
from utils.extra import cleanup

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("medium")


"""Functions that classification metrics"""


def calculate_classification_performance(config, device, acc_mode):
    out_fp = f'{config.results_dir}/classification_performance.csv'

    # Load results if available
    results_df = pd.read_csv(out_fp) if os.path.exists(out_fp) else pd.DataFrame()

    # Check if results are already logged
    if config.run_name in results_df.get('run_name',  pd.Series(dtype=str)).tolist():
        logging.info(f'The results of {config.run_name} are already saved.')
        return

    # Dataloader
    dataloader = setup_dataloader(config)
    dataloader.setup('test')

    # Model and trainer
    trainer = pl.Trainer()
    model = setup_model(config, testing_mode=True, acc_mode=acc_mode).to(device)

    # Get the results
    results = trainer.test(model, dataloaders=dataloader.test_dataloader())[0]
    save_results(config, out_fp, results, results_df)
    del model, dataloader
    cleanup()


def save_results(config, out_fp, results, results_df):
    # Append run name
    results['run_name'] = config.run_name

    # Append results to the read in csv
    results_df = pd.concat([results_df, pd.DataFrame([results])])
    results_df = results_df.sort_values(by="run_name", key=natsort_keygen())

    # Save results
    results_df.to_csv(out_fp, header=True, index=False)
    logging.info(f'{config.run_name} model performance saved to: {out_fp}')

