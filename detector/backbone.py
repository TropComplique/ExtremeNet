import torch.nn as nn


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        num_filters = 32
        self.beginning = nn.Sequential(
            nn.Conv2d(3, num_filters, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU6(inplace=True)
        )
        previous_filters = num_filters

        strides_and_filters = [
            (1, 64),
            (2, 128), (1, 128),  # c2
            (2, 256), (1, 256),  # c3
            (2, 512), (1, 512), (1, 512), (1, 512), (1, 512), (1, 512),  # c4
            (2, 1024), (1, 1024)  # c5
        ]

        layers = []
        for stride, num_filters in strides_and_filters:
            layers.append(DepthwisePointwise(previous_filters, num_filters, stride))
            previous_filters = num_filters

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, h, w],
                it represents RGB images with pixel values in the range [0, 1].
        Returns:
            a dict with float tensors.
        """
        features = {}

        x = 2.0 * x - 1.0
        x = self.beginning(x)

        x = self.layers[:3](x)
        features['c2'] = x  # stride 4

        x = self.layers[3:5](x)
        features['c3'] = x  # stride 8

        x = self.layers[5:11](x)
        features['c4'] = x  # stride 16

        x = self.layers[11:](x)
        features['c5'] = x  # stride 32

        return features


class DepthwisePointwise(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwisePointwise, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels, eps=1e-3),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
        Returns:
            a float tensor with shape [b, out_channels, h // stride, w // stride],
            where h and w are divisible by the stride.
        """
        x = self.layers(x)
        return x
