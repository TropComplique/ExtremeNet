import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from detector.architecture import Architecture
from detector.losses import focal_loss, regression_loss


class Trainer:
    def __init__(self, num_steps):
        """
        Arguments:
            num_steps: an integer.
        """
        self.network = Architecture(num_outputs=5 + 10)
        self.optimizer = optim.Adam(self.network.parameters(), lr=3e-4, weight_decay=1e-6)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_steps, eta_min=1e-7)

    def get_losses(self, images, labels):
        """
        Arguments:
            images: a float tensor with shape [b, 3, h, w],
                it represents RGB images with
                pixel values in the range [0, 1].
            labels: a dict with float tensors.
        Returns:
            a dict with float tensors of shape [].
        """

        split = torch.split(labels['masks'], [1, 1], dim=1)
        segmentation_mask, loss_mask = split
        # they have shapes [b, 1, h/4, w/4]

        x, enriched_features = self.network(images)
        heatmaps, offsets = torch.split(x, [5, 10], dim=1)
        # they have shapes [b, 5, h/4, w/4] and [b, 10, h/4, w/4]

        losses = focal_loss(labels, heatmaps, alpha=2.0, beta=4.0)  # shape [b, h/4, w/4]
        heatmap_loss = (loss_mask.squeeze(1) * losses).sum([1, 2]).mean(0)

        losses = regression_loss(labels, offsets)  # shape [b, h/4, w/4]
        offset_loss = (loss_mask.squeeze(1) * losses).sum([1, 2]).mean(0)

        # this is additional supervision using segmentation masks
        additional_loss = torch.tensor(0.0, device=x.device)

        for level in ['2', '3', '4', '5']:

            p = enriched_features['p' + level][:, 0]
            # it has shape [b, h/s, w/s], where s ** level

            losses = F.mse_loss(p, segmentation_mask.squeeze(1), reduction='none')
            additional_loss += (loss_mask.squeeze(1) * losses).sum([1, 2]).mean(0)

            # 2x downsampling
            segmentation_mask = F.interpolate(segmentation_mask, scale_factor=0.5, mode='bilinear', align_corners=False)
            loss_mask = F.interpolate(loss_mask, scale_factor=0.5, mode='bilinear', align_corners=False)

        return {
            'heatmap_loss': heatmap_loss,
            'offset_loss': offset_loss,
            'additional_loss': additional_loss
        }

    def train_step(self, images, labels):

        losses = self.get_losses(images, labels)
        total_loss = 1e-4 * losses['additional_loss'] + losses['heatmap_loss']
        total_loss += 2.0 * losses['offset_loss']

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        losses.update({'total_loss': total_loss})
        return losses

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def evaluate(self, images, labels):
        """
        Evaluation is on batches of size 1.
        """

        with torch.no_grad():
            losses = self.get_losses(images, labels)

        return losses
