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
        self.bn1 = nn.BatchNorm2d(outfeat[0])
        self.conv2 = nn.Conv2d(outfeat[0], outfeat[1], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outfeat[1])
        self.conv3 = nn.Conv2d(outfeat[1], outfeat[2], kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outfeat[2])
        self.relu = nn.ReLU(inplace=True)
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


class ResNet_MSBP(nn.Module):
    def __init__(self, exp_mode, nr_classes, freeze=False):
        super(ResNet_MSBP, self).__init__()
        self.exp_mode = exp_mode
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

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer5 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)

        self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, return_indices=True)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))  # Global Max Pooling

        if self.exp_mode == 'ResNet':
            self.classifier = nn.Linear(2048, self.nr_classes)
        elif self.exp_mode == 'ResNet_add':
            # quy version, BUG, likely wrong, there is no ReLU
            self.fc_scale = nn.Linear(256, 1024)
            self.classifier = nn.Linear(1024, self.nr_classes)
        elif 'ResNet_conv' in self.exp_mode:
            # quy version, BUG, likely wrong, there is no ReLU
            self.conv_scale = nn.Conv2d(256, 512, kernel_size=1, bias=False)
            self.classifier = nn.Linear(512, self.nr_classes)
        elif 'ResNet_concat' in self.exp_mode:
            # quy version, BUG, likely wrong, there is no ReLU
            self.fc_scale = nn.Linear(1280, 1024)
            self.classifier = nn.Linear(1024, self.nr_classes)
        else: #SBP
            self.classifier = nn.Linear(1024, self.nr_classes)

            self.conv_u = nn.ModuleList()
            self.conv_v = nn.ModuleList()

            in_ch = 256
            self.conv_u.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, 512, 1, stride=1, padding=0, bias=True),
                    nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(512, eps=1e-5, momentum=0.9),
                    nn.ReLU(inplace=True),
                )
            )

            self.conv_v.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, 512, 1, stride=1, padding=0, bias=True),
                    nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(512, eps=1e-5, momentum=0.9),
                    nn.ReLU(inplace=True),
                )
            )

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

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        # Trying adding different weights on features of top-down and
        # bottom-up.
        # weights = [0.8, 0.2]
        # return F.upsample(weights[0]*x, size=(H,W), mode='bilinear') + weights[1]*y

        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, imgs):
        def scale_to(x, size):
            return nn.functional.interpolate(x, size=tuple(size),
                                             mode='bilinear', align_corners=True)

        def extract_feat(imgs):
            d1 = self.relu(self.bn1(self.conv1(imgs)))
            d1 = self.maxpool(d1)
            d2 = self.layer1(d1)
            d3 = self.layer2(d2)
            d4 = self.layer3(d3)
            d5 = self.layer4(d4)

            d1 = self.latlayer5(d1)
            d2 = self.latlayer4(d2)
            d3 = self.latlayer3(d3)
            d4 = self.latlayer2(d4)
            d5 = self.latlayer1(d5)

            return [d1, d2, d3, d4, d5]

        # feature extractor only
        if self.exp_mode == 'ResNet':
            with torch.no_grad():
                feat = extract_feat(imgs)[-1]
                feat = self.gap(feat)  # NOTE: Global Average Pool
        elif self.exp_mode == 'ResNet_MSBP':
            fpn_feat = extract_feat(imgs)
            levels_feat_list = []
            for feat in fpn_feat:
                scale_feat = scale_to(feat, list(fpn_feat[2].shape[2:]))
                levels_feat_list.append(scale_feat)

            scale_feat = torch.stack(levels_feat_list, dim=4)
            avg = torch.mean(scale_feat, dim=4, keepdim=False)
            avgs = [avg, avg, avg, avg, avg]
            avgs = torch.stack(avgs, dim=4)
            # print("mean: ", time.time() - start)
            binary_code = torch.sub(scale_feat, avgs)
            binary_code = torch.where(binary_code >= 0,
                                      torch.tensor(1).to('cuda'), torch.tensor(0).to('cuda'))

            binary_code[:, :, :, :, 1] = torch.mul(binary_code[:, :, :, :, 1],
                                                   torch.tensor(2).to('cuda'))
            binary_code[:, :, :, :, 2] = torch.mul(binary_code[:, :, :, :, 2],
                                                   torch.tensor(4).to('cuda'))
            binary_code[:, :, :, :, 3] = torch.mul(binary_code[:, :, :, :, 3],
                                                   torch.tensor(8).to('cuda'))
            binary_code[:, :, :, :, 4] = torch.mul(binary_code[:, :, :, :, 4],
                                                   torch.tensor(16).to('cuda'))
            decimal_code = torch.sum(binary_code, dim=-1, dtype=torch.float32)

            u = self.conv_u[0](decimal_code)
            v = self.conv_v[0](avg)
            feat = torch.cat((u, v), dim=1)
            feat = self.gmp(feat)  # NOTE: Global Max Pool

        else:
            if 'zm' in self.exp_mode:
                fpn_feat = extract_feat(imgs)
                levels_feat_list = []
                for feat in fpn_feat:
                    scale_feat = scale_to(feat, list(fpn_feat[2].shape[2:]))
                    scale_feat = torch.squeeze(scale_feat)
                    levels_feat_list.append(scale_feat)

                scale_feat = torch.stack(levels_feat_list, dim=4)
                avg = torch.mean(scale_feat, dim=4, keepdim=False)
                avgs = [avg, avg, avg, avg, avg]
                avgs = torch.stack(avgs, dim=4)
                subtract = torch.sub(scale_feat, avgs)
                scale_feat = [self.gap(subtract[..., i]).view(scale_feat.size(0), -1) for i in range(subtract.shape[-1])]
                scale_feat = torch.stack(scale_feat, dim=-1)
            else:
                fpn_feat = extract_feat(imgs)
                levels_feat_list = []
                for feat in fpn_feat:
                    scale_feat = scale_to(feat, list(fpn_feat[2].shape[2:]))
                    scale_feat = self.gmp(scale_feat)
                    scale_feat = torch.squeeze(scale_feat)
                    levels_feat_list.append(scale_feat)

                scale_feat = torch.stack(levels_feat_list, dim=-1)
                if len(scale_feat.shape) != 3:
                    scale_feat = torch.unsqueeze(scale_feat, 0)

            if 'ResNet_add' in self.exp_mode:
                # sum across scale, element wise
                feat = torch.sum(scale_feat, -1)
                feat = self.fc_scale(feat)
            elif 'ResNet_concat' in  self.exp_mode:
                # flatten all features
                feat = scale_feat.view(scale_feat.size(0), -1)
                feat = self.fc_scale(feat)
            elif 'ResNet_conv' in self.exp_mode:
                feat = torch.unsqueeze(scale_feat, 3)
                feat = self.conv_scale(feat)
                feat = self.gmp(feat)

        out = feat.view(feat.size(0), -1)
        out = self.classifier(out)
        return out


def resnet_msbp(exp_mode, nr_classes, pretrained=True, progress=True):
    model = ResNet_MSBP(exp_mode=exp_mode, nr_classes=nr_classes)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model

# def _test():
#     import os
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#     net = resnet_msbp(exp_mode='ResNet_MSBP', nr_classes=4).cuda()
#     p = net(torch.randn(4, 3, 512, 512).cuda())
#     print(p.size())
#
#     # model = net.cuda()
#     # summary(model, (3, 224, 224))
# _test()
