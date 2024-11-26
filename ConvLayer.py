import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    
    def __init__(self, in_channels=1, kernel_size=5, out_channels = 32, stride=1, padding=0):
        '''Constructs the ConvLayer with specified input and output sizes.
           param in_channels: input depth of an image, default value = 1 for grayscale images
           param kernel_size: size of the convolution kernel, default value = 9
           param stride: stride of the convolution, default value = 1
           param padding: zero-padding added to both sides of the input, default value = 0
        '''
        super(ConvLayer, self).__init__()
        
        # Define convolutional and pooling layers
        self.conv1 = nn.Conv2d(
            in_channels, 32, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            32, out_channels, kernel_size=kernel_size, 
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
        # First convolution and activation
        x = F.relu(self.conv1(x))
        
        # Apply pooling
        x = self.pool(x)
        
        # Second convolution and activation
        x = F.relu(self.conv2(x))
        
        return x

# Test the layer
if __name__ == '__main__':
    import torch

    # Create an instance of the ConvLayer
    conv_layer = ConvLayer(in_channels=1, kernel_size=9, stride=1, padding=0)

    # Dummy input tensor: batch size = 4, 1 channel (grayscale), image size 64x64
    dummy_input = torch.randn(4, 1, 64, 64)

    # Pass through the ConvLayer
    output = conv_layer(dummy_input)

    # Print output shape
    print("Output Shape:", output.shape)
