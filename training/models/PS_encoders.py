import logging

from torch import nn
import torch
from torchvision import models

"""PS encoders (Resnet, Single and 3-layer)"""


class PSResNet(nn.Module):
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

        self.pretrained = pretrained
        if self.pretrained:
            self._load_weights()

        self.net.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.net.fc = nn.Identity()

    def _load_weights(self):
        if self.pretrained != 'ssl':
            self.net.conv1 = nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

            if self.pretrained == 'dino':
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

            logging.info(f'PS {self.base_model} encoder loaded with MoCo weights')
            del checkpoint
            torch.cuda.empty_cache()

    def forward(self, x):
        z = self.net(x)
        return z


class PSPreEncoderSingle(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=128, kernel_size=4, stride=4),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.net(x)


class PSPreEncoder3Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)