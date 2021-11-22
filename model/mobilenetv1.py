import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# import matplotlib.pyplot as plt

from collections import OrderedDict


class MobileNet(nn.Module):
    def __init__(self, num_classes):
        super(MobileNet, self).__init__()
        self.nr_classes = num_classes

        self.extractor = nn.Sequential(
            self.conv_bn(3, 32, 2),
            self.conv_dw(32, 64, 1),
            self.conv_dw(64, 128, 2),
            self.conv_dw(128, 128, 1),
            self.conv_dw(128, 256, 2),
            self.conv_dw(256, 256, 1),
            self.conv_dw(256, 512, 2),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 1024, 2),
            self.conv_dw(1024, 1024, 1)
        )


        self.AvgPool = nn.AvgPool2d(7)
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(1024, self.nr_classes)


    def conv_bn(self, inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    def conv_dw(self, inp, oup, stride):  # dw means depth wise
        return nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    def forward(self, imgs):
            x = self.extractor(imgs)
            x = self.gap(x)
            x = x.view(-1, 1024)
            x = self.classifier(x)
            return x

