import torch.nn as nn
import torch.nn.functional as F


def focal_loss(y, y_hat, num_objects, alpha, beta):
    """
    Arguments:
        y, y_hat: float tensors with shape [b, c, h, w].
        num_objects: a long tensor with shape [].
    Return:
        a float tensor with shape [].
    """
    is_extreme_point = torch.equal(y, 1.0)  # binary tensor
    losses = F.binary_cross_entropy_with_logits(
        input=y_hat, target=is_extreme_point.long(),
        reduction='none'
    )  # shape [b, c, h, w]

    weights = torch.where(
        is_extreme_point, torch.pow(1.0 - y_hat, alpha),
        torch.pow(1.0 - y, beta) * torch.pow(y_hat, alpha)
    )

    batch_size = y.size(0)
    normalizer = num_objects * batch_size
    return torch.sum(weights * losses, [0, 1, 2, 3])/normalizer
