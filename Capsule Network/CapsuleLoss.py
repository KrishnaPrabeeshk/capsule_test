import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleLoss(nn.Module):
    
    def __init__(self):
        '''Constructs a CapsuleLoss module.'''
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(reduction='sum')  # Cumulative loss
    
    def forward(self, x, labels, images, reconstructions):
        '''Defines how the loss compares inputs.
           param x: digit capsule outputs
           param labels: the correct class labels for each image (binary: cat or dog)
           param images: the original input images (e.g., cats or dogs)
           param reconstructions: reconstructed image data
           return: weighted margin and reconstruction loss, averaged over the batch size
           '''
        batch_size = x.size(0)

        ## Calculate the margin loss ##
        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))
        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)
        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        ## Calculate the reconstruction loss ##
        # Flatten both reconstructions and images to match dimensions
        reconstructions = reconstructions.view(batch_size, -1)  # Shape: [batch_size, 4096]
        images = images.view(batch_size, -1)  # Shape: [batch_size, 4096]
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        # Return a weighted, summed loss, averaged over the batch size
        return (margin_loss + 0.0005 * reconstruction_loss) / batch_size
