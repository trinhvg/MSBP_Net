
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, infeat, outfeat, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(infeat, outfeat[0], kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(outfeat[0])
        self.conv2 = nn.Conv2d(outfeat[0], outfeat[1], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(outfeat[1])
        self.conv3 = nn.Conv2d(outfeat[1], outfeat[2], kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(outfeat[2])
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, nr_classes, freeze=True):
        super(ResNet, self).__init__()
        self.nr_classes = nr_classes

        self.freeze = freeze
        # NOTE: using name to load tf chkpts easier
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_res_block(64, [64, 64, 256], 3, stride=1)
        self.layer2 = self._make_res_block(256, [128, 128, 512], 4, stride=2)
        self.layer3 = self._make_res_block(512, [256, 256, 1024], 6, stride=2)
        self.layer4 = self._make_res_block(1024, [512, 512, 2048], 3, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))  # Global Max Pooling

        self.classifier = nn.Linear(2048, self.nr_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def _make_res_block(self, infeat, outfeat, nr_blocks, stride=1):
        downsample = None
        if stride != 1 or infeat != outfeat[-1]:
            downsample = nn.Sequential(
                nn.Conv2d(infeat, outfeat[-1], kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outfeat[-1]),
            )

        layers = []
        layers.append(Bottleneck(infeat, outfeat, stride, downsample))
        for _ in range(1, nr_blocks):
            layers.append(Bottleneck(outfeat[-1], outfeat))

        return nn.Sequential(*layers)

    def forward(self, imgs):

        def extract_feat(imgs):
            with torch.no_grad():
                d1 = self.relu(self.bn1(self.conv1(imgs)))
                d2 = self.maxpool(d1)
                d2 = self.layer1(d2)
                d3 = self.layer2(d2)
            d4 = self.layer3(d3)
            d5 = self.layer4(d4)
            return [d1, d2, d3, d4, d5]

        # feature extractor only
        feat = extract_feat(imgs)[-1]
        feat = self.gap(feat)  # NOTE: Global Average Pool

        out = feat.view(feat.size(0), -1)
        out = self.classifier(out)
        return out

def resnet(exp_mode, nr_classes, pretrained=True, progress=True):
    model = ResNet(exp_mode=exp_mode, nr_classes=nr_classes)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model

