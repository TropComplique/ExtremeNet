import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(labels, predictions, alpha, beta):
    """
    Arguments:
        labels: a dict with the following keys
            'heatmaps': a float tensor with shape [b, c, h, w],
            'num_boxes': a long tensor with shape [b].
        predictions: a float tensor with shape [b, c, h, w].
        alpha, beta: float numbers.
    Returns:
        a float tensor with shape [b, h, w].
    """

    # notation like in the paper
    y = labels['heatmaps']
    y_hat = predictions

    is_extreme_point = y.eq(1.0)
    # binary tensor with shape [b, c, h, w]

    losses = F.binary_cross_entropy_with_logits(
        input=y_hat, reduction='none',
        target=is_extreme_point.float()
    )  # shape [b, c, h, w]

    weights = torch.where(
        is_extreme_point, torch.pow(1.0 - y_hat, alpha),
        torch.pow(1.0 - y, beta) * torch.pow(y_hat, alpha)
    )  # shape [b, c, h, w]

    b = y.size(0)  # batch size
    normalizer = labels['num_boxes'].view(b, 1, 1).float() + 1.0
    return (weights * losses).sum(1)/normalizer


def regression_loss(labels, predictions):
    """
    Arguments:
        labels: a dict with the following keys
            'heatmaps': a float tensor with shape [b, c, h, w],
            'offsets': a float tensor with shape [b, 2 * c, h, w],
            'num_boxes': a long tensor with shape [b].
        predictions: a float tensor with shape [b, 2 * c, h, w].
    Returns:
        a float tensor with shape [b, h, w].
    """

    is_extreme_point = labels['heatmaps'].eq(1.0)
    # binary tensor with shape [b, c, h, w]

    # note that is_extreme_point.sum([1, 2, 3])
    # must be equal to labels['num_boxes']

    weights = is_extreme_point.repeat(1, 2, 1, 1).float()
    # shape [b, 2 * c, h, w]

    losses = F.smooth_l1_loss(predictions, labels['offsets'], reduction='none')
    # shape [b, 2 * c, h, w]

    b = predictions.size(0)  # batch size
    normalizer = labels['num_boxes'].view(b, 1, 1).float() + 1.0
    return (weights * losses).sum(1)/normalizer
