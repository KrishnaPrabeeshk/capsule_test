import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvDecoder(nn.Module):
    def __init__(self, digit_caps_out_dim=8, digit_caps_count=32, initial_channels=16):
        """
        Constructs a convolutional decoder for image reconstruction using only PrimaryCaps output.
        :param primary_caps_dim: Dimension of PrimaryCaps vectors
        :param primary_caps_count: Number of PrimaryCaps
        :param initial_channels: Channels used in the first ConvTranspose2d layer
        """
        super(ConvDecoder, self).__init__()

        # Calculate input size
        self.decoder_digit_caps_input_dim = digit_caps_out_dim * digit_caps_count

        # Fully connected layer to project to spatial tensor
        self.fc = nn.Sequential(
            nn.Linear(self.decoder_digit_caps_input_dim, initial_channels * 8 * 8),
            nn.ReLU(inplace=True)
        )

        # Transpose convolution layers for upsampling
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(initial_channels, 32, kernel_size=4, stride=2, padding=1),  # Upsample to 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Upsample to 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # Upsample to 64x64
            nn.Sigmoid()  # Normalize pixel intensities to [0, 1]
        )

    def forward(self, digit_caps_output):
        """
        Forward pass of the decoder.
        :param primary_caps_output: Output from PrimaryCaps layer
        :return: Reconstructed image
        """
       # Flatten PrimaryCaps output
        digit_caps_flat = digit_caps_output.contiguous().view(digit_caps_output.size(0), -1)

        # Fully connected layer
        x = self.fc(digit_caps_flat)
        x = x.view(x.size(0), -1, 8, 8)  # Reshape to spatial tensor

        # Transpose convolution for reconstruction
        reconstructions = self.deconv_layers(x)
        return reconstructions


if __name__ == '__main__':
    # Instantiate the ConvDecoder with appropriate parameters
    decoder = ConvDecoder(digit_caps_out_dim=8, digit_caps_count=32, initial_channels=16)

    # Create a dummy input tensor simulating the output of PrimaryCaps
    # Shape: (batch_size, primary_caps_count, primary_caps_dim)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 32, 8)  # 32 capsules, each with 8 dimensions

    # Perform a forward pass
    reconstructions = decoder(dummy_input)

    # Print the outputs
    print("Reconstructed Images Shape:", reconstructions.shape)  # Expect: (batch_size, 1, 64, 64)
