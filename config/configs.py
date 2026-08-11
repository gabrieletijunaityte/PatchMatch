import glob
import json
import os
import logging


class Configs:
    """
    A class to read and store the configurations for training.

    """
    def __init__(self, config_path, create=True):
        self.config_path = config_path
        self.create = create

        self._read_and_parse()

    def _read_and_parse(self):
        """
        Parses given config path
        Returns
        -------
        None
        """
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        try:
            # Configuration info
            self.run_name = config['run_name']
            self.note = config.get('note', None)

            # PS and S2 scene directories
            self.PS_dir = config['Scenes']['PS_directory']
            self.S2_dir = config['Scenes']['S2_directory']
            if self.create:
                os.makedirs(self.PS_dir, exist_ok=True)
                os.makedirs(self.S2_dir, exist_ok=True)

            # Path to database
            self.DB_path = config['DB_path']

            # Resampling
            self.resample = config['Pre_tiles']['resample']

            # Pre-tile pre-processing info
            self.pt_dir = config['Pre_tiles']['directory']
            if self.create: os.makedirs(self.pt_dir, exist_ok=True)
            self.pt_template = config['Pre_tiles']['template']
            self.pt_size = config['Pre_tiles']['size']
            self.right_pt_size = self.pt_size if self.resample else int(self.pt_size / (10 / 3))
            self.pt_json = config['Pre_tiles']['json']
            self.pt_json_path = os.path.join(self.pt_dir, self.pt_json)
            self.pt_test_json = config['Pre_tiles']['test_json']
            self.pt_test_json_path = os.path.join(self.pt_dir, self.pt_test_json)

            # Tile size
            self.tile_size = config['Tiles']['size']
            self.right_tile_size = config['Tiles']['right_size']

            # Normalisation
            self.normalise = config['Tiles']['normalise']
            self.bandwise = config['Tiles'].get('bandwise', None)

            # Architecture
            self.PS_encoder = config['Architecture']['PS_encoder']
            self.S2_encoder = config['Architecture']['S2_encoder']
            self.backbone = config['Architecture']['Backbone']
            self.projection_head = config['Architecture'].get('Projection_head', None)
            self.hidden_dim = config['Architecture'].get('hidden_dim', None)

            # Loss
            self.loss = config['Architecture']['Loss']
            self.loss_margin = config['Architecture'].get('Loss_margin', None)
            self.loss_temp = config['Architecture'].get('Loss_temperature', None)

            # Similarity
            self.sim_name = config['Architecture']['Similarity']
            self.sim_mlp_dim = config['Architecture'].get('Similarity_MLP_dim', None)
            self.similarity_threshold = config['Architecture']['Similarity_threshold']

            # Pre-trained initialisation
            self.gen_pretrained = config['Training'].get('general_pretraining', None)
            self.ssl_pretrained = config['Training'].get('ssl_pretraining', None)
            self.freeze = config['Training'].get('freeze', None)

            # Positive pair probabilities
            self.train_positive_prob = config['Training']['train_positive_probability']
            self.test_validate_positive_prob = config['Training'].get('test_validate_positive_probability', 1)

            # Augmentations
            self.augmentation_prob = config['Training']['augmentation_probability']
            self.augmentations = config['Training']['augmentations']

            # Training settings
            self.lr = config['Training']['learning_rate']
            self.epochs = config['Training']['epochs']

            # Batch sizes
            self.batch_size = config['Training']['batch_size']
            self.validate_batch_size = config['Training'].get('validate_batch', self.batch_size)
            self.test_batch_size = config['Testing'].get('test_batch', self.batch_size)

            # WandB and weight path
            self.project_name = config['Training']['wandb_project']
            self.checkpoint_dir = config['Training']['checkpoint_dir']
            if self.create: os.makedirs(self.checkpoint_dir, exist_ok=True)

            # Predictions
            self.pred_dir = config['Testing']['Prediction_dir']
            if self.create: os.makedirs(self.pred_dir, exist_ok=True)
            self.top_ks = config['Testing']['top_ks']
            self.pred_path = os.path.join(self.pred_dir, self.run_name, 'global_pred_matrix.csv')
            if self.create: os.makedirs(os.path.join(self.pred_dir, self.run_name), exist_ok=True)
            self.pred_fig_parent_dir = config['Testing']['Prediction_figures']
            self.pred_fig_dir = os.path.join(self.pred_fig_parent_dir, self.run_name)
            if self.create: os.makedirs(self.pred_fig_dir, exist_ok=True)
            self.results_dir = config['Testing']['Results_dir']
            if self.create: os.makedirs(self.results_dir, exist_ok=True)

        except KeyError as e:
            logging.info(f"Missing key in config file: {e}")
            raise e

        # Additional
        # Data split polygons
        self.test_poly_dir = config['Pre_tiles'].get('test_polygon_dir', "data/test_areas/")
        self.validate_poly_dir = config['Pre_tiles'].get('validate_polygon_dir', "data/validate_areas/")

        # Information about scenes
        self.PS_converted = config.get('PS_converted', True)
        self.S2_available = config.get('S2_available', True)
        self.GEE_project = config.get('GEE_project', "thesis-patchmatchnn")

        # Directories for annotations
        self.ndi_dir = config.get('directories', {}).get('ndi', "intermediate/ndi")
        self.styles_dir = config.get('directories', {}).get('styles', "data/styles")
        self.python_dir = config.get('directories', {}).get('python', None)
        self.annotations_dir = config.get('directories', {}).get('annotations', "data/annotations", )
        self.projects_dir = config.get('directories', {}).get('projects', "intermediate/qgis")

        # Info about annotation steps
        self.ndi_calculated = config.get('ndi_calculated', True)
        self.qgis_projects_created = config.get('qgis_projects_created', True)
        self.S2_search_radius = config.get('S2_search_radius', 10)

        # SSL implementation: data
        self.ssl_project = config.get('SSL', {}).get("wandb_project", None)
        self.ssl_pairs_file = config.get('SSL', {}).get('pairs_path', None)
        self.ssl_pairs = os.path.join(self.pt_dir, self.ssl_pairs_file) if self.ssl_pairs_file else None

        # SSL implementation: pre-training settings
        self.ssl_epochs = config.get('SSL', {}).get('epochs', None)
        self.ssl_temp = config.get('SSL', {}).get('temperature', None)
        self.ssl_hidden_dim = config.get('SSL', {}).get('hidden_dim', None)
        self.ssl_lr = config.get('SSL', {}).get('learning_rate', None)
        self.ssl_weight_decay = config.get('SSL', {}).get('weight_decay', None)
        self.ssl_batch_size = config.get('SSL', {}).get('batch_size', None)
        self.ssl_augmentations = config.get('SSL', {}).get('ssl_augmentations', None)
        self.ssl_mode = config.get('SSL', {}).get('mode', None)
        self.ssl_saved_epoch = config.get('SSL', {}).get('saved_epoch', None)

    def set_up_models(self, branch=None, pretrained=None):
        from training.models.PS_encoders import PSResNet, PSPreEncoderSingle, PSPreEncoder3Layer
        from training.models.S2_encoders import  S2ResNet, S2PreEncoderSingle, S2PreEncoder3Layer
        from training.models.backbones import  ResNetBackbone
        from training.models.projection_heads import ProjectionHeadFNN, ProjectionHeadLinear
        import torch.nn as nn

        """
        Set up encoders based on the config file.
        Parameters
        ----------
        branch - which branch encoders to return, PS, S2 or all.
        pretrained - should encoders be initialised pretrained (from SSL4EO)

        Returns
        -------
        PS_encoder, S2_encoder and bacbone

        """
        # Change the parameter for the model initialisation from the pre-trained weights
        pretrained = pretrained or 'ssl' if self.ssl_pretrained else self.gen_pretrained

        models = {'PSResNet18': lambda: PSResNet('resnet18', pretrained=pretrained),
            'PSResNet50': lambda: PSResNet('resnet50', pretrained=pretrained),
            'PSResNet50_dino': lambda: PSResNet('resnet50_dino', pretrained=pretrained),
            'PSPreEncoderSingle': lambda: PSPreEncoderSingle(), 'PSPreEncoder3Layer': lambda: PSPreEncoder3Layer(),

            'S2ResNet18': lambda: S2ResNet('resnet18', pretrained=pretrained),
            'S2ResNet50': lambda: S2ResNet('resnet50', pretrained=pretrained),
            'S2ResNet50_dino': lambda: S2ResNet('resnet50_dino', pretrained=pretrained),
            'S2PreEncoderSingle': lambda: S2PreEncoderSingle(), 'S2PreEncoder3Layer': lambda: S2PreEncoder3Layer(),

            "ResNet18": lambda: ResNetBackbone('resnet18', pretrained=pretrained), }
        try:
            PS_encoder = models[self.PS_encoder]() if branch != "S2" else None
            S2_encoder = models[self.S2_encoder]() if branch != "PS" else None

            if self.backbone:
                backbone = models[self.backbone]()
            else:
                backbone = None

            # Individual projection heads
            if self.projection_head == "linear" and branch not in ["S2", "PS"] and backbone is None:
                PS_encoder = nn.Sequential(PS_encoder, ProjectionHeadLinear(input_dim=PS_encoder.out_dim, output_dim=128))
                S2_encoder = nn.Sequential(S2_encoder, ProjectionHeadLinear(input_dim=S2_encoder.out_dim, output_dim=128))
            if self.loss == "Relational_NN_loss":
                PS_encoder = nn.Sequential(PS_encoder, ProjectionHeadFNN(input_dim=PS_encoder.out_dim, hidden_dim=1024, output_dim=300))
                S2_encoder = nn.Sequential(S2_encoder, ProjectionHeadFNN(input_dim=S2_encoder.out_dim, hidden_dim=1024, output_dim=300))

            return PS_encoder, S2_encoder, backbone
        except:
            raise KeyError(f'There is no model called {self.PS_encoder} or {self.S2_encoder}.')

    def get_ssl_pairs(self, platform=None):
        """Gets train/validate split of PS and S2 (separately) tiles for SSL weight refinement"""
        with open(self.ssl_pairs) as pairs_json:
            pairs = json.load(pairs_json)
        if platform is None:
            return pairs
        else:
            try:
                return pairs[platform]
            except:
                raise KeyError()

    def get_train_pairs(self):
        """Gets training pairs (paths to tiles) from the json files"""
        try:
            with open(self.pt_json_path) as pairs_json:
                pairs = json.load(pairs_json)
            return pairs
        except:
            return None

    def get_transforms_strategy(self, strategy=None, ssl=True):
        """Compiles augmentation strategies (mild, medium, harsh and shuffling"""
        from training.datamodule.augmentations import random_zoom_effect, adjust_contrast, adjust_brightness, shuffle_bands
        from torchvision.transforms import v2
        import torch
        if ssl:
            strategy = self.ssl_augmentations
        else:
            strategy = strategy
            if strategy is None:
                return None, None

        if "mild" in strategy:
            contrast_transforms_PS = v2.Compose([
                v2.RandomApply(torch.nn.ModuleList([v2.ElasticTransform(alpha=30)]), p=0.5),
                v2.Lambda(lambda img: random_zoom_effect(img, self.tile_size - 100, self.pt_size + 100)),
                v2.RandomRotation((-20, 20)),
                v2.RandomApply(torch.nn.ModuleList([v2.GaussianBlur(kernel_size=11, sigma=(0.5, 2))]), p=0.5),
                v2.GaussianNoise(sigma=0.002, clip=True)])

            contrast_transforms_S2 = v2.Compose([
                v2.RandomApply(torch.nn.ModuleList([v2.ElasticTransform(alpha=30)]), p=0.5),
                v2.Lambda(lambda img: random_zoom_effect(img, self.right_tile_size - 10, self.right_pt_size + 10)),
                v2.RandomRotation((-20, 20)),
                v2.RandomApply(torch.nn.ModuleList([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))]), p=0.5),
                v2.GaussianNoise(sigma=0.001, clip=True)])
        elif 'mid' in strategy:
            contrast_transforms_PS = v2.Compose([
                v2.RandomApply(torch.nn.ModuleList([v2.ElasticTransform(alpha=30)]), p=0.5),
                v2.Lambda(lambda img: random_zoom_effect(img, self.tile_size - 150, self.pt_size + 150)),
                v2.RandomRotation((-45, 45)), v2.GaussianNoise(sigma=0.002, clip=True),
                v2.RandomApply(torch.nn.ModuleList([v2.GaussianBlur(kernel_size=11, sigma=(0.5, 2))]), p=0.5),
                v2.RandomApply(torch.nn.ModuleList([v2.Lambda(lambda img: adjust_brightness(img, "mid"))]), p=0.5),
                v2.RandomApply(torch.nn.ModuleList([v2.Lambda(lambda img: adjust_contrast(img))]), p=0.5), ])

            contrast_transforms_S2 = v2.Compose([
                v2.RandomApply(torch.nn.ModuleList([v2.ElasticTransform(alpha=30)]), p=0.5),
                v2.Lambda(lambda img: random_zoom_effect(img, self.right_tile_size - 30, self.right_pt_size + 30)),
                v2.RandomRotation((-45, 45)), v2.GaussianNoise(sigma=0.001, clip=True),
                v2.RandomApply(torch.nn.ModuleList([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))]), p=0.5),
                v2.RandomApply(torch.nn.ModuleList([
                                                       v2.Lambda(lambda img: adjust_brightness(img, "mid", True))]), p=0.5),
                v2.RandomApply(torch.nn.ModuleList([v2.Lambda(lambda img: adjust_contrast(img))]), p=0.5), ])
        elif "harsh" in strategy:
            contrast_transforms_PS = v2.Compose([
                v2.RandomApply(torch.nn.ModuleList([v2.ElasticTransform(alpha=30)]), p=0.5),
                v2.Lambda(lambda img: random_zoom_effect(img, self.tile_size - 150, self.pt_size + 150)),
                v2.RandomRotation((-90, 90)), v2.RandomApply([v2.GaussianBlur(kernel_size=11, sigma=(0.5, 2))], p=0.5),
                v2.GaussianNoise(sigma=0.002, clip=True),
                v2.RandomApply(torch.nn.ModuleList([v2.GaussianBlur(kernel_size=11, sigma=(0.5, 2))]), p=0.5),
                v2.RandomApply(torch.nn.ModuleList([v2.Lambda(lambda img: adjust_brightness(img, "all"))]), p=0.5),
                v2.RandomApply(torch.nn.ModuleList([v2.Lambda(lambda img: adjust_contrast(img))]), p=0.5), ])

            contrast_transforms_S2 = v2.Compose([
                v2.RandomApply(torch.nn.ModuleList([v2.ElasticTransform(alpha=30)]), p=0.5),
                v2.Lambda(lambda img: random_zoom_effect(img, self.right_tile_size - 30, self.right_pt_size + 30)),
                v2.RandomRotation((-90, 90)), v2.GaussianNoise(sigma=0.001, clip=True),
                v2.RandomApply(torch.nn.ModuleList([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))]), p=0.5),
                v2.RandomApply(torch.nn.ModuleList([
                                                       v2.Lambda(lambda img: adjust_brightness(img, "all", True))]), p=0.5),
                v2.RandomApply(torch.nn.ModuleList([v2.Lambda(lambda img: adjust_contrast(img))]), p=0.5), ])
        elif "shuffle" not in strategy:
            raise NotImplementedError(f"Augmentation strategy {strategy} not implemented")
        else:
            contrast_transforms_S2 = None
            contrast_transforms_PS = None

        if "shuffle" in strategy:
            if contrast_transforms_S2 and contrast_transforms_PS:
                contrast_transforms_PS = v2.Compose([contrast_transforms_PS,
                    v2.RandomApply([v2.Lambda(lambda img: shuffle_bands(img))], p=0.5)])

                contrast_transforms_S2 = v2.Compose([contrast_transforms_S2,
                    v2.RandomApply([v2.Lambda(lambda img: shuffle_bands(img))], p=0.5)])
            else:
                contrast_transforms_PS = v2.RandomApply([v2.Lambda(lambda img: shuffle_bands(img))], p=0.5)

                contrast_transforms_S2 = v2.RandomApply([v2.Lambda(lambda img: shuffle_bands(img))], p=0.5)

        return contrast_transforms_PS, contrast_transforms_S2

    def get_pre_trained_summary(self):
        """Gives more readable summary of pre-training procedures in the configuration"""
        if self.gen_pretrained:
            if self.ssl_pretrained:
                if len(self.ssl_mode) == 1:
                    pre_trained = f'{self.gen_pretrained} & {self.ssl_mode} SSL - frozen {self.freeze if self.freeze else "nothing"}'
                else:
                    pre_trained = f'{self.gen_pretrained} & individual SSL - frozen {self.freeze if self.freeze else "nothing"}'
            else:
                pre_trained = f'{self.gen_pretrained} - frozen {self.freeze if self.freeze else "nothing"}'
        elif self.ssl_pretrained:
            if len(self.ssl_mode) == 1:
                pre_trained = f'{self.ssl_mode} SSL - frozen {self.freeze if self.freeze else "nothing"}'
            else:
                pre_trained = f'Individual SSL - frozen {self.freeze if self.freeze else "nothing"}'
        else:
            pre_trained = None

        return pre_trained

    @property
    def ssl_path(self, epoch=None):
        """Puts together path to the SSL pre-trained weights"""
        if not self.ssl_saved_epoch:
            return glob.glob(os.path.join(self.checkpoint_dir, self.run_name + '*SSL*'))
        else:
            paths = glob.glob(os.path.join(self.checkpoint_dir, f"SSL_*_ep={self.ssl_saved_epoch}.ckpt"))
            if len(paths) == 2:
                return paths
            else:
                raise ValueError("Incorrect SSL paths")

    @property
    def pre_trained_path(self, acc_mode=None):
        """Puts together path to the trained model weights."""
        return glob.glob(os.path.join(self.checkpoint_dir, self.run_name + f'-{acc_mode}*'))

    def save(self):
        """ Saves updated configuration to the original json file."""
        config = {
            'run_name': self.run_name,
            'note': self.note,
            'Scenes': {
                'PS_directory': self.PS_dir,
                'S2_directory': self.S2_dir,
            },
            'DB_path': self.DB_path,
            'Pre_tiles': {
                'directory': self.pt_dir,
                'template': self.pt_template,
                'size': self.pt_size,
                'json': self.pt_json,
                'test_json': self.pt_test_json,
                'resample': self.resample,
                'test_polygon_dir': self.test_poly_dir,
                'validate_polygon_dir': self.validate_poly_dir,
            },

            'Tiles': {
                'size': self.tile_size,
                'right_size': self.right_tile_size,
                'normalise': self.normalise,
                'bandwise': self.bandwise,
            },

            'Architecture': {
                'PS_encoder': self.PS_encoder,
                'S2_encoder': self.S2_encoder,
                'Backbone': self.backbone,
                'Projection_head': self.projection_head,
                'hidden_dim': self.hidden_dim,
                'Loss': self.loss,
                'Loss_margin': self.loss_margin,
                'Loss_temperature': self.loss_temp,
                'Similarity': self.sim_name,
                'Similarity_threshold': self.similarity_threshold,
            },

            'Training': {
                'general_pretraining': self.gen_pretrained,
                'ssl_pretraining': self.ssl_pretrained,
                'freeze': self.freeze,
                'train_positive_probability': self.train_positive_prob,
                'test_validate_positive_probability': self.test_validate_positive_prob,
                'learning_rate': self.lr,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'validate_batch': self.validate_batch_size,
                'wandb_project': self.project_name,
                'checkpoint_dir': self.checkpoint_dir,
                'augmentation_probability': self.augmentation_prob,
                'augmentations': self.augmentations,
            },

            'Testing': {
                'test_batch': self.test_batch_size,
                'Prediction_dir': self.pred_dir,
                'top_ks': self.top_ks,
                'Prediction_figures': self.pred_fig_parent_dir,
                "'Results_dir": self.results_dir
            },

            'SSL': {
                'wandb_project': self.ssl_project,
                'pairs_path': self.ssl_pairs_file,
                'epochs': self.ssl_epochs,
                'temperature': self.ssl_temp,
                'hidden_dim': self.ssl_hidden_dim,
                'learning_rate': self.ssl_lr,
                'weight_decay': self.ssl_weight_decay,
                'batch_size': self.ssl_batch_size,
                'mode': self.ssl_mode,
                'ssl_augmentations': self.ssl_augmentations,
                'saved_epoch': self.ssl_saved_epoch
            },
            'PS_converted': self.PS_converted,
            'S2_available': self.S2_available,
            'GEE_project': self.GEE_project,

            'directories': {
                'ndi': self.ndi_dir,
                'styles': self.styles_dir,
                'python': self.python_dir,
                'annotations': self.annotations_dir,
                'projects': self.projects_dir,
            },

            'ndi_calculated': self.ndi_calculated,
            'qgis_projects_created': self.qgis_projects_created,
            'S2_search_radius': self.S2_search_radius,
        }

        # Save to JSON file
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)