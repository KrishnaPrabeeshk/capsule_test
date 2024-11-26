import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    
    def __init__(self, input_vector_length=16, input_capsules=2, hidden_dim=512, output_dim=128*128):
        '''Constructs a series of linear layers + activations.
           param input_vector_length: dimension of input capsule vector
           param input_capsules: number of capsules in previous layer (e.g., 2 for binary classes)
           param hidden_dim: dimensions of hidden layers
           param output_dim: dimensions of the output image, e.g., 128*128 for grayscale
           '''
        super(Decoder, self).__init__()
        
        # Calculate input dimension
        input_dim = input_vector_length * input_capsules
        
        # Define linear layers + activations
        self.linear_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),        # First hidden layer
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),   # Second hidden layer
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, output_dim),   # Output layer for grayscale images
            nn.Sigmoid()
        )

    def forward(self, x):
        '''Defines the feedforward behavior.
           param x: the input vectors from the DigitCaps layer
           return: reconstructed images and class scores, y
           '''
        # Compute vector lengths, indicating class probabilities
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        
        # Identify the capsule with the maximum vector length
        _, max_length_indices = classes.max(dim=1)
        
        # Create a sparse class matrix for binary classes
        sparse_matrix = torch.eye(2, device=x.device)
        y = sparse_matrix.index_select(dim=0, index=max_length_indices)
        
        # Mask out capsules based on selected class
        x = x * y[:, :, None]
        
        # Flatten image into a vector shape (batch_size, vector_dim) using reshape
        flattened_x = x.reshape(x.size(0), -1)
        
        # Generate reconstructed image vectors
        reconstructions = self.linear_layers(flattened_x)
        
        return reconstructions, y
