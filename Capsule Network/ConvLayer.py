import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9, stride=1, padding=0):
        '''Constructs the ConvLayer with specified input and output sizes.
           param in_channels: input depth of an image, default value = 1 for grayscale images
           param out_channels: output depth of the convolutional layer, default value = 256
           param kernel_size: size of the convolution kernel, default value = 9
           param stride: stride of the convolution, default value = 1
           param padding: zero-padding added to both sides of the input, default value = 0
        '''
        super(ConvLayer, self).__init__()
        
        # Define a convolutional layer with the specified parameters
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        '''Defines the feedforward behavior and calculates output dimensions.
           param x: the input to the layer; an input image tensor
           return: a ReLU-activated convolutional layer output
        '''
        # Apply convolution and ReLU activation
        features = F.relu(self.conv(x))
        
        # Calculate output dimensions based on input dimensions
        input_height = x.shape[2]  # Height of the input image
        input_width = x.shape[3]   # Width of the input image

        # Output dimensions formula for convolutional layers
        output_height = (input_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        output_width = (input_width - self.kernel_size + 2 * self.padding) // self.stride + 1

        # Optionally, print the output dimensions for debugging
        # print(f"Output dimensions after ConvLayer: {output_height}x{output_width}x{features.shape[1]}")
        
        return features