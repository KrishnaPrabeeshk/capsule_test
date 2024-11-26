import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleLoss(nn.Module):
    def __init__(self, reconstruction_loss_weight=0.0005):
        """
        Capsule Loss: Combines margin loss and reconstruction loss for image reconstruction.
        :param reconstruction_loss_weight: Weight for the reconstruction loss.
        """
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss_weight = reconstruction_loss_weight

    def forward(self, digit_caps_output, images, reconstructions):
        """
        Compute the combined margin loss and reconstruction loss.
        :param digit_caps_output: Output from DigitCaps layer (batch_size, num_capsules, capsule_dim).
        :param images: Original input images (batch_size, channels, height, width).
        :param reconstructions: Reconstructed images from the decoder (batch_size, channels, height, width).
        :return: Combined loss.
        """
        # Compute capsule vector lengths (presence probability of features)
        v_c = torch.sqrt((digit_caps_output ** 2).sum(dim=2))  # Capsule vector lengths: (batch_size, num_capsules)

        # Margin loss for capsule vectors
        positive_loss = F.relu(0.9 - v_c) ** 2  # Encourages feature presence
        negative_loss = F.relu(v_c - 0.1) ** 2  # Penalizes irrelevant features
        margin_loss = (positive_loss + 0.5 * negative_loss).mean()  # Average over capsules and batch

        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(
            reconstructions.view(reconstructions.size(0), -1),  # Flatten reconstructed images
            images.view(images.size(0), -1),                   # Flatten original images
            reduction='mean'
        )

        # Total loss: margin loss + weighted reconstruction loss
        total_loss = margin_loss + self.reconstruction_loss_weight * reconstruction_loss
        return total_loss
