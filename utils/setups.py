import glob
import logging
import os

"""Functions to setup models, dataloaders, wandb tracker"""

def setup_model(config, weight_path=None, testing_mode=False, acc_mode=None, **kwargs):
    import torch
    """
    Loads the model based on configuration file and chosen parameters
    ----------
    config - configuration object
    weight_path - specific path for weights (if needed).
    testing_mode - load non-trained or pre-trained model.
    acc_mode - which accuracy mode (used for monitoring the weights) to use.
    kwargs

    Returns
    -------
    model object

    """
    from training.proposed_model import ProposedModel

    # Check if chosen acc_mode exists
    if acc_mode and acc_mode not in ['validate_acc', 'validate_top_3_acc', 'validate_top_1_acc', 'validate_top_5_acc', 'validate_top_10_acc']:
        raise ValueError('Accuracy mode must be one of: validate_acc,  validate_top_1_acc, validate_top_3_acc, validate_top_5_acc, validate_top_10_acc')

    if testing_mode:
        # Load model with pre-trained weights for model testing
        PS_encoder, S2_encoder, shared_backbone = config.set_up_models(pretrained=False)
        if weight_path is not None and os.path.isfile(weight_path):
            checkpoint_path = weight_path
        else:
            checkpoint_path = glob.glob(f'output/weights/{config.run_name}-{acc_mode}*.ckpt')
            if len(checkpoint_path) == 1:
                checkpoint_path = checkpoint_path[0]
            elif len(checkpoint_path) > 1:
                raise Exception(f'Multiple checkpoints found for {config.run_name}, go delete unnecessary files.')
            else:
                raise Exception(f'No checkpoint found, go train a model for {config.run_name}')

        model = ProposedModel.load_from_checkpoint(
            checkpoint_path,
            left_encoder=PS_encoder,
            right_encoder=S2_encoder,
            shared_backbone=shared_backbone,
            loss_mode=config.loss,
            hidden_dim=config.hidden_dim,
            loss_margin=config.loss_margin,
            similarity_mode=config.sim_name,
            similarity_threshold=config.similarity_threshold,
            lr=config.lr,
            max_epochs=config.epochs,
            projection_head=config.projection_head,
            loss_temp=config.loss_temp)
        logging.info(f'Loaded the model from {checkpoint_path}.')
    else:
        # Load non-trained model
        if config.ssl_pretrained:
            PS_encoder, S2_encoder, shared_backbone = config.set_up_models(pretrained=False)
        else:
            PS_encoder, S2_encoder, shared_backbone = config.set_up_models()

        model = ProposedModel(
            left_encoder=PS_encoder,
            right_encoder=S2_encoder,
            shared_backbone=shared_backbone,
            loss_mode=config.loss,
            loss_margin=config.loss_margin,
            hidden_dim=config.hidden_dim,
            similarity_mode=config.sim_name,
            similarity_threshold=config.similarity_threshold,
            lr=config.lr,
            max_epochs=config.epochs,
            projection_head=config.projection_head,
            freeze=config.freeze,
            loss_temp=config.loss_temp,
            ssl_path=config.ssl_path if config.ssl_pretrained else None,
            batch_size=config.batch_size,
            val_batch_size=config.validate_batch_size,
            test_batch_size=config.test_batch_size, )
        logging.info(f'A new model initialised based on {config.run_name}.')
    model = model.to(torch.float32)
    return model


def setup_dataloader(config):
    from training.datamodule.dataset import DebrisDataModule
    transforms, right_transforms = config.get_transforms_strategy(strategy=config.augmentations, ssl=False)

    dataloader = DebrisDataModule(
        pairs=config.get_train_pairs(),
        tile_size=config.tile_size,
        right_tile_size=config.right_tile_size,
        train_positive_prob=config.train_positive_prob,
        test_validate_positive_prob=config.test_validate_positive_prob,
        batch_size=config.batch_size,
        val_batch_size=config.validate_batch_size,
        test_batch_size=config.test_batch_size,
        trans_prob=config.augmentation_prob,
        trans=transforms,
        right_trans=right_transforms,
        normalise=config.normalise,
        bandwise=config.bandwise,
        num_workers=16)
    return dataloader


def set_up_logger(config, train_params=None, ssl=False):
    from lightning.pytorch.loggers import WandbLogger
    # Set up the logger
    # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb
    if ssl:
        wandb_logger = WandbLogger(project=config.ssl_project, name=config.run_name, log_model="all")

        wandb_logger.log_hyperparams({'Note': config.note, "PS encoder": config.PS_encoder,
                                      "S2 encoder": config.S2_encoder, "Shared backbone": config.backbone,
                                      'PS size': config.tile_size, 'S2 size': config.right_tile_size,
                                      'Batch size': config.ssl_batch_size,
                                      'Normalised': 'Bandwise-normalised' if config.normalise and config.bandwise else 'Normalised' if config.normalise else 'Scaled', })
    else:
        wandb_logger = WandbLogger(project=config.project_name, name=config.run_name, log_model="all")

        pre_trained = config.get_pre_trained_summary()

        wandb_logger.log_hyperparams({'Note': config.note, "PS encoder": config.PS_encoder,
                                      "S2 encoder": config.S2_encoder, "Shared backbone": config.backbone,
                                      'Epochs': config.epochs, 'Batch size': config.batch_size,
                                      'Validation batch size': config.validate_batch_size,
                                      'Test batch size': config.test_batch_size, 'PS size': config.tile_size,
                                      'S2 size': config.right_tile_size, 'Augmentations': config.augmentations,
                                      'Trainable parameters': train_params, 'Pretrained': pre_trained,
                                      'Normalised': 'Bandwise-normalised' if config.normalise and config.bandwise else 'Normalised' if config.normalise else 'Scaled',
                                      'Similarity threshold': config.similarity_threshold})

    return wandb_logger
