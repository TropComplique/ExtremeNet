import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import MobileNet
from .fpn import FPN


class Architecture(nn.Module):
    def __init__(self, num_outputs):
        super(Architecture, self).__init__()

        self.backbone = MobileNet()
        self.fpn = FPN(depth=128)
        self.phi_subnet = PhiSubnet(in_channels=128, depth=64, num_copies=4)
        self.end = nn.Sequential(
            nn.Conv2d(4 * 64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_outputs, 1)
        )

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
            level = str(i + 2)
            p = enriched_features['p' + level]
            upsampled_features.append(self.phi_subnet(p, i + 2))

        x = torch.cat(upsampled_features, dim=1)
        x = self.end(x)
        return x, enriched_features


class PhiSubnet(nn.Module):
    def __init__(self, in_channels, depth, num_copies):
        super(PhiSubnet, self).__init__()

        self.layers = nn.ModuleList([
            ConditionalBatchNorm(in_channels, num_copies),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, depth, 3, padding=1, bias=False),
            ConditionalBatchNorm(depth, num_copies),
            nn.ReLU(inplace=True),
            nn.Conv2d(depth, depth, 3, padding=1, bias=False),
            ConditionalBatchNorm(depth, num_copies),
            nn.ReLU(inplace=True),
        ])

    def forward(self, x, level):
        """
        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
            level: an integer. Possible values are [2, 3, 4, 5].
        Returns:
            a float tensor with shape [b, depth, upsample * h, upsample * w],
            where upsample = 2**(level - 2).
        """
        for i in range(8):
            if i in [0, 3, 6]:
                x = self.layers[i](x, torch.tensor([level - 2]))
            else:
                x = self.layers[i](x)
        x = F.interpolate(x, scale_factor=2**(level - 2), mode='bilinear', align_corners=True)
        return x


class ConditionalBatchNorm(nn.Module):

    def __init__(self, d, n):
        super(ConditionalBatchNorm, self).__init__()

        self.bn = nn.BatchNorm2d(d, affine=False)
        self.embedding = nn.Embedding(n, 2 * d)
        self.embedding.weight.data[:, :d] = 1.0
        self.embedding.weight.data[:, d:] = 0.0

    def forward(self, x, i):
        """
        Arguments:
            x: a float tensor with shape [b, d, h, w].
            i: a long tensor with shape [1].
        Returns:
            a float tensor with shape [b, d, h, w].
        """
        x_normalized = self.bn(x)  # shape [b, d, h, w]
        params = self.embedding(i)  # shape [1, 2 * d]
        gamma, beta = torch.split(params, x.size(1), dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(2)  # shape [1, d, 1, 1]
        beta = beta.unsqueeze(2).unsqueeze(2)  # shape [1, d, 1, 1]
        return gamma * x_normalized + beta
