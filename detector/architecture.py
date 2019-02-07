import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
import math
from .backbone import MobileNet
from .fpn import FPN


class Architecture(nn.Module):
    def __init__(self, num_outputs):
        super(Architecture, self).__init__()

        self.backbone = MobileNet()
        self.fpn = FPN(depth=64)
        self.phi_subnets = nn.ModuleList([
            PhiSubnet(in_channels=64, depth=32, level=i+2)
            for i in range(4)
        ])
        self.end = nn.Sequential(
            nn.Conv2d(4 * 32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_outputs, 1)
        )

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self.apply(weights_init)
        p = 0.01  # probability of foreground
        torch.nn.init.constant_(self.end[3].bias, -math.log((1.0 - p) / p))
        # sigmoid(-log((1 - p) / p)) = p

    def forward(self, x):
        """
        It assumed that h and w are divisible by 32.

        Arguments:
            x: a float tensor with shape [b, 3, h, w],
                it represents RGB images with pixel values in the range [0, 1].
        Returns:
            x: a float tensor with shape [b, num_outputs, h/4, w/4].
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
        self.level = level  # possible values are [2, 3, 4, 5]

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
        Returns:
            a float tensor with shape [b, depth, upsample * h, upsample * w],
            where upsample = 2**(level - 2).
        """
        x = self.layers(x)
        x = F.interpolate(x, scale_factor=2**(self.level - 2), mode='bilinear', align_corners=True)
        return x
