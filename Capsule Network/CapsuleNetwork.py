import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvLayer import ConvLayer
from PrimaryCaps import PrimaryCaps
from ClassCapsules import DigitCaps
from Decoder import Decoder

class CapsuleNetwork(nn.Module):
    
    def __init__(self):
        '''Constructs a complete Capsule Network.'''
        super(CapsuleNetwork, self).__init__()
        
        # Initialize ConvLayer for grayscale images
        self.conv_layer = ConvLayer(in_channels=1)  # Grayscale images have 1 channel
        
        # Initialize PrimaryCaps
        self.primary_capsules = PrimaryCaps(
            num_capsules=8,       # Number of capsule types
            in_channels=256,      # Must match ConvLayer's out_channels
            out_channels=32,      # Number of units per capsule type
            kernel_size=9,
            stride=2
        )
        
        # Initialize DigitCaps (in_capsules is determined dynamically)
        self.digit_capsules = DigitCaps(
            num_capsules=2,       # For binary classification (cats vs. dogs)
            in_dim=8,             # Capsule dimension from PrimaryCaps
            out_dim=16,           # Output capsule dimension
            routing_iterations=3  # Number of routing iterations
        )

        # Initialize Decoder
        self.decoder = Decoder(
            input_vector_length=16,  # Should match out_dim from DigitCaps
            input_capsules=2,        # Number of output capsules from DigitCaps
            hidden_dim=512,
            output_dim=64*64       # Flattened size of the grayscale image
        )
                
    def forward(self, images):
        x = self.conv_layer(images)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        reconstructions, y = self.decoder(x)
        reconstructions = reconstructions.view(-1, 1, 64, 64)
        return x, reconstructions, y
