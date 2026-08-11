import torch.nn as nn
from torchvision import models
import torch
import logging

"""Shared architecture definitions"""


class ResNetBackbone(nn.Module):
    def __init__(self, backbone_model, pretrained=False):
        super().__init__()

        # Base model name
        self.base_model = backbone_model

        if backbone_model == "resnet18":
            self.net = models.resnet18(weights=None)
            self.out_dim = 512
            self.weights = 'weights/B13_rn18_moco_0099_ckpt.pth'
        elif backbone_model == "resnet50":
            self.net = models.resnet50(weights=None)
            self.out_dim = 2048
            self.weights = 'weights/B13_rn50_moco_0099_ckpt.pth'
        else:
            raise NotImplementedError

        # Pre-trained weight initialisation
        self.pretrained = pretrained
        if self.pretrained:
            self.load_weights()

        # Replacing first and last layers
        self.net.conv1 = nn.Conv2d(128, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.net.fc = nn.Identity()

    def load_weights(self):
        """Loading pre-trained weights"""
        if self.pretrained == 'moco':
            self.net.conv1 = nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

            checkpoint = torch.load(self.weights, map_location=torch.device('cpu'))["state_dict"]

            checkpoint = {k.replace("module.encoder_q.", ""): v for k, v in checkpoint.items() if
                          k.startswith("module.encoder_q.")}

            missing_keys, unexpected_keys = self.net.load_state_dict(checkpoint, strict=False)
            if missing_keys:
                logging.info(f"The following keys are missing from the pretrained model: {missing_keys}")
            if unexpected_keys:
                logging.info(f"The following keys are unexpected from the pretrained model: {unexpected_keys}")

            logging.info(f'Backbone {self.base_model} encoder loaded with MoCo weights')
            del checkpoint
            torch.cuda.empty_cache()

    def forward(self, x):
        z = self.net(x)
        return z
