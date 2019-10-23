import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """
    This is an implementation
    of feature pyramid network.
    """
    def __init__(self, depth):
        super(FPN, self).__init__()

        # for mobilenet outputs
        input_filters = [1024, 512, 256, 128]

        self.reduce_dimension = nn.ModuleList([
            nn.Conv2d(num_filters, depth, 1)
            for num_filters in input_filters
        ])
        self.smoothen = nn.ModuleList([
            nn.Conv2d(depth, depth, 3, padding=1, bias=False)
            for _ in range(4)
        ])

    def forward(self, features):
        """
        Arguments:
            features: a dict with float tensors.
                It has keys ['c2', 'c3', 'c4', 'c5'].
            Returns:
                a dict with float tensors.
                It has keys ['p2', 'p3', 'p4', 'p5'].
        """

        x = self.reduce_dimension[0](features['c5'])
        p5 = self.smoothen[0](x)
        enriched_features = {'p5': p5}

        # top-down path
        for i, level in enumerate(['4', '3', '2'], 1):
            lateral = self.reduce_dimension[i](features['c' + level])
            x = F.interpolate(x, scale_factor=2, mode='nearest') + lateral
            enriched_features['p' + level] = self.smoothen[i](x)

        return enriched_features
