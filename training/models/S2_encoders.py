import logging

from torch import nn as nn
import torch
from torchvision import models

"""S2 encoders (Resnet, Single and 3-layer)"""


class S2ResNet(nn.Module):
    def __init__(self, base_model, pretrained=False):
        super().__init__()
        self.base_model = base_model
        if base_model == "resnet18":
            self.net = models.resnet18(weights=None)
            self.out_dim = 512
            self.weights = 'weights/B13_rn18_moco_0099_ckpt.pth'
        elif base_model == "resnet50":
            self.net = models.resnet50(weights=None)
            self.out_dim = 2048
            self.weights = 'weights/B13_rn50_moco_0099_ckpt.pth'
        elif base_model == "resnet50_dino":
            self.net = models.resnet50(weights=None)
            self.out_dim = 2048
            self.weights = 'weights/B13_rn50_dino_0099.pth'
        else:
            raise NotImplementedError

        self.net.conv1 = nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.pretrained = pretrained
        if self.pretrained:
            self.load_weights()

        self.net.fc = nn.Identity()

    def load_weights(self):
        if self.pretrained != 'ssl':
            if self.pretrained == "dino":
                checkpoint = torch.load(self.weights, map_location=torch.device('cpu'))

            else:
                checkpoint = torch.load(self.weights, map_location=torch.device('cpu'))["state_dict"]

            checkpoint = {k.replace("module.encoder_q.", ""): v for k, v in checkpoint.items() if
                          k.startswith("module.encoder_q.")}

            missing_keys, unexpected_keys = self.net.load_state_dict(checkpoint, strict=False)
            if missing_keys:
                logging.info(f"The following keys are missing from the pretrained model: {missing_keys}")
            if unexpected_keys:
                logging.info(f"The following keys are unexpected from the pretrained model: {unexpected_keys}")

            logging.info(f'S2 {self.base_model} encoder loaded with MoCo weights')
            del checkpoint
            torch.cuda.empty_cache()

    def forward(self, x):
        z = self.net(x)
        return z


class S2PreEncoderSingle(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=128, kernel_size=1, stride=1)
        )

    def forward(self, x):
        return self.net(x)


class S2PreEncoder3Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(13, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
