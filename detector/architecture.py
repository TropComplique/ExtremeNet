import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from detector.backbone import MobileNet
from detector.fpn import FPN


class Architecture(nn.Module):
    """
    This architecture is inspired by paper
    "MultiPoseNet: Fast Multi-Person Pose Estimation using Pose Residual Network"
    (https://arxiv.org/abs/1807.04067)
    """
    def __init__(self, num_outputs):
        super(Architecture, self).__init__()

        fpn_depth = 64
        phi_depth = 32

        self.backbone = MobileNet()
        self.fpn = FPN(fpn_depth)

        self.phi_subnets = nn.ModuleList([
            PhiSubnet(fpn_depth, phi_depth, level=i + 2)
            for i in range(4)
        ])
        self.end = nn.Sequential(
            nn.Conv2d(4 * phi_depth, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_outputs, 1)
        )

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

        # initialize all weights
        self.apply(weights_init)

        p = 0.01  # probability of foreground
        init.constant_(self.end[3].bias, -math.log((1.0 - p) / p))
        # because sigmoid(-log((1 - p) / p)) = p

    def forward(self, x):
        """
        It assumed that h and w are divisible by 32.

        Arguments:
            x: a float tensor with shape [b, 3, h, w],
                it represents RGB images with
                pixel values in the range [0, 1].
        Returns:
            x: a float tensor with shape [b, num_outputs, h / 4, w / 4].
            enriched_features: a dict with float tensors.
        """

        features = self.backbone(x)
        enriched_features = self.fpn(features)

        upsampled_features = []
        for i in range(4):
            p = enriched_features[f'p{i + 2}']
            upsampled_features.append(self.phi_subnets[i](p))

        x = torch.cat(upsampled_features, dim=1)
        x = self.end(x)
        return x, enriched_features


class PhiSubnet(nn.Module):

    def __init__(self, in_channels, depth, level):
        super(PhiSubnet, self).__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, depth, 3, padding=1, bias=False),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True),
            nn.Conv2d(depth, depth, 3, padding=1, bias=False),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True),
        )

        self.level = level
        # possible values are [2, 3, 4, 5]

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
        Returns:
            a float tensor with shape [b, depth, upsample * h, upsample * w],
            where upsample = 2**(level - 2).
        """

        x = self.layers(x)
        # it has shape [b, depth, h, w]

        upsample = 2 ** (self.level - 2)
        x = F.interpolate(x, scale_factor=upsample, mode='bilinear')
        return x
