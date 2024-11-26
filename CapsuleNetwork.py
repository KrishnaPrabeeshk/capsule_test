import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvLayer import ConvLayer
from PrimaryCaps import PrimaryCaps
from ClassCapsules import DigitCaps
# from Decoder import Decoder
from Decoder_conv import ConvDecoder

class CapsuleNetwork(nn.Module):
    def __init__(self):
        '''Constructs a complete Capsule Network.'''
        super(CapsuleNetwork, self).__init__()

        # Initialize ConvLayer for grayscale images
        self.conv_layer = ConvLayer(in_channels=1, out_channels=32, kernel_size=5)  # Grayscale images have 1 channel

        # Initialize PrimaryCaps
        self.primary_capsules = PrimaryCaps(
            num_capsules=64,       # Number of capsule types
            in_channels=32,       # Must match ConvLayer's out_channels
            out_channels=8,      # Number of units per capsule type
            kernel_size=9,
            stride=2
        )
        # output params = 8*

        # Initialize DigitCaps
        self.digit_capsules = DigitCaps(
            num_capsules=32,       # For binary classification
            in_dim=648,             # Capsule dimension from PrimaryCaps
            out_dim=16,           # Output capsule dimension
            routing_iterations=3  # Number of routing iterations
        )

        # Initialize Decoder
        self.decoder = ConvDecoder(
            digit_caps_out_dim=16,
            digit_caps_count=32
        )

    def forward(self, images):
        # Pass through layers
        x = self.conv_layer(images)
        primary_caps_output = self.primary_capsules(x)
        digit_caps_output = self.digit_capsules(primary_caps_output)

        # Compute the class probabilities (length of capsule vectors)
        class_scores = (digit_caps_output ** 2).sum(dim=-1) ** 0.5  # Capsule lengths as class scores
        class_scores = F.softmax(class_scores, dim=-1)

        # Mask the capsules based on the predicted class
        _, max_length_indices = class_scores.max(dim=1)
        sparse_mask = torch.eye(digit_caps_output.size(1), device=digit_caps_output.device)
        mask = sparse_mask.index_select(dim=0, index=max_length_indices)

        # Pass both PrimaryCaps and DigitCaps outputs to the decoder
        reconstructions = self.decoder(digit_caps_output)

        return digit_caps_output, reconstructions, class_scores

    





if __name__ == '__main__':
    # Instantiate the CapsuleNetwork
    model = CapsuleNetwork()

    # Create a dummy input tensor simulating grayscale images
    # Shape: (batch_size, 1, height, width)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 64, 64)  # Grayscale images (1 channel)

    # Perform a forward pass
    capsule_output, reconstructions, class_scores = model(dummy_input)

    # Print the outputs
    print("Capsule Output Shape:", capsule_output.shape)  # Shape depends on DigitCaps output
    print("Reconstructed Images Shape:", reconstructions.shape)  # Expect: (batch_size, 1, 64, 64)
    print("Class Scores Shape:", class_scores.shape)  # Expect: (batch_size, 2) for binary classification

