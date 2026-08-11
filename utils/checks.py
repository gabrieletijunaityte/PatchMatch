import glob
import logging

"""Functions to check if models were (pre)trained already, if yes, to not re-train"""


def check_if_trained(config, acc_mode='validate_top_5_acc'):
    checkpoint_path = glob.glob(f'output/weights/{config.run_name}-{acc_mode}*.ckpt')

    if len(checkpoint_path) == 1:
        logging.info(f'Model for {config.run_name} is already trained.')
        return True
    elif len(checkpoint_path) > 1:
        raise Exception(f'Multiple checkpoints found for {config.run_name}, go delete unnecessary files.')
    else:
        return False


def check_if_pretrained(config):
    if config.ssl_mode in ["sequential", "combined"]:
        checkpoint_path = glob.glob(f'output/weights/{config.run_name}_{config.ssl_mode}-SSL-*.ckpt')
        if len(checkpoint_path) == 1:
            logging.info(f'Model for {config.run_name} is already pre-trained.')
            return True
    else:
        checkpoint_path = glob.glob(f'output/weights/{config.run_name}_{config.ssl_mode[0]}-SSL-*.ckpt')
        checkpoint_path = checkpoint_path + (
            glob.glob(f'output/weights/{config.run_name}_{config.ssl_mode[1]}-SSL-*.ckpt'))
        if len(checkpoint_path) == 2:
            logging.info(f'Model for {config.run_name} is already pre-trained.')
            return True

    if len(checkpoint_path) == 0:
        return False
    else:
        raise Exception(f'Multiple checkpoints found for {config.run_name}, go delete unnecessary files.')
