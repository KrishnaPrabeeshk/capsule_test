import torch.nn.functional as F
import torch.nn as nn

class DecoderLoss(nn.Module):
    def __init__(self):
        """
        A loss function for image reconstruction tasks.
        Uses Mean Squared Error (MSE) to compare the original and reconstructed images.
        """
        super(DecoderLoss, self).__init__()

    def forward(self, reconstructions, original_images):
        """
        Compute the reconstruction loss.
        :param reconstructions: Output from the decoder (batch_size, 1, 64, 64).
        :param original_images: Original input images (batch_size, 1, 64, 64).
        :return: Reconstruction loss value.
        """
        # Flatten the images for MSE computation
        reconstructions_flat = reconstructions.view(reconstructions.size(0), -1)
        original_images_flat = original_images.view(original_images.size(0), -1)

        # Compute Mean Squared Error
        loss = F.mse_loss(reconstructions_flat, original_images_flat, reduction='mean')
        return loss
