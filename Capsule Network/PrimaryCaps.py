import torch
import torch.nn as nn
import torch.nn.functional as F

class PrimaryCaps(nn.Module):
    
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, stride=2, padding=0):
        '''Constructs a set of convolutional capsule layers to be used in 
           creating capsule output vectors.
           param num_capsules: number of types of capsules to create (default: 8)
           param in_channels: input depth of features from ConvLayer (default: 256)
           param out_channels: number of capsule units per type (default: 32)
           param kernel_size: size of the convolution kernel (default: 9)
           param stride: stride of the convolution (default: 2)
           param padding: zero-padding added to both sides of the input (default: 0)
        '''
        super(PrimaryCaps, self).__init__()
        
        # Create a list of convolutional layers for each type of capsule
        self.num_capsules = num_capsules
        self.out_channels = out_channels
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=kernel_size, stride=stride, padding=padding)
            for _ in range(num_capsules)])
        
        # Save the parameters for dynamic output dimension calculation
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def forward(self, x):
        '''Defines the feedforward behavior.
           param x: the input; features from the ConvLayer
           return: a set of normalized, capsule output vectors
        '''
        # Get batch size of inputs
        batch_size = x.size(0)
        
        # Dynamically calculate output dimensions based on the input dimensions
        input_height = x.shape[2]  # Height of the input feature map
        input_width = x.shape[3]   # Width of the input feature map

        # Output dimensions formula for convolutional layers
        output_height = (input_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        output_width = (input_width - self.kernel_size + 2 * self.padding) // self.stride + 1

        # Reshape each capsule output to match the calculated output dimensions
        u = []
        for capsule in self.capsules:
            # Apply the capsule's convolutional layer to the input
            capsule_output = capsule(x)  # Shape: [batch_size, out_channels, output_height, output_width]
            # Flatten the capsule output and reshape
            capsule_output = capsule_output.view(batch_size, -1, 1)
            u.append(capsule_output)
        
        # Concatenate the capsule outputs along the capsule dimension
        # Shape after cat: [batch_size, num_units, num_capsules]
        u = torch.cat(u, dim=2)

        # Apply squashing to the stack of vectors
        u_squash = self.squash(u)
        return u_squash
    
    def squash(self, input_tensor):
        '''Squashes an input Tensor so it has a magnitude between 0-1.
           param input_tensor: a stack of capsule inputs, s_j
           return: a stack of normalized, capsule output vectors, v_j
        '''
        squared_norm = (input_tensor ** 2).sum(dim=2, keepdim=True)
        scale = squared_norm / (1 + squared_norm)  # Normalization coefficient
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm + 1e-9)    
        return output_tensor
