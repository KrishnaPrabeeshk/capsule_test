import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitCaps(nn.Module):
    
    def __init__(self, num_capsules=2, in_capsules=None, in_dim=8, out_dim=16, routing_iterations=3):
        '''Initializes the DigitCaps layer.
           param num_capsules: number of output capsules (e.g., 2 for binary classification)
           param in_capsules: number of input capsules from PrimaryCaps (to be set dynamically)
           param in_dim: dimension of each input capsule vector (from PrimaryCaps)
           param out_dim: dimension of each output capsule vector
           param routing_iterations: number of routing iterations to perform
        '''
        super(DigitCaps, self).__init__()

        self.num_capsules = num_capsules
        self.in_capsules = in_capsules  # Will be set dynamically
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.routing_iterations = routing_iterations

        # Weights will be initialized dynamically once in_capsules is known
        self.W = None

    def forward(self, u):
        '''Defines the forward pass.
           param u: the input tensor from PrimaryCaps, shape [batch_size, in_capsules, in_dim]
           return: a set of normalized, capsule output vectors
        '''
        batch_size = u.size(0)
        if self.W is None:
            # Initialize weights dynamically based on in_capsules
            self.in_capsules = u.size(1)
            self.W = nn.Parameter(torch.randn(1, self.in_capsules, self.num_capsules, self.out_dim, self.in_dim).to(u.device))

        # Add batch dimension to W
        W = self.W.repeat(batch_size, 1, 1, 1, 1)  # Shape: [batch_size, in_capsules, num_capsules, out_dim, in_dim]

        # u.permute(0, 2, 1)
        # Unsqueeze u for matrix multiplication
        u = u.unsqueeze(2).unsqueeze(-1)  # Shape: [batch_size, in_capsules, 1, in_dim, 1]

        # Compute u_hat: predicted output vectors
        u_hat = torch.matmul(W, u)  # Shape: [batch_size, in_capsules, num_capsules, out_dim, 1]
        u_hat = u_hat.squeeze(-1)   # Shape: [batch_size, in_capsules, num_capsules, out_dim]

        # Initialize routing logits b_ij to zero
        b_ij = torch.zeros(batch_size, self.in_capsules, self.num_capsules, 1).to(u.device)

        # Perform dynamic routing
        v_j = self.dynamic_routing(u_hat, b_ij)

        return v_j  # Shape: [batch_size, num_capsules, out_dim]
    
    def dynamic_routing(self, u_hat, b_ij):
        '''Performs dynamic routing between capsules.
           param u_hat: predicted output vectors from the lower-level capsules
           param b_ij: initial routing logits
           return: output capsule vectors after routing
        '''
        batch_size = u_hat.size(0)

        for iteration in range(self.routing_iterations):
            # Apply softmax to b_ij along the num_capsules dimension
            c_ij = F.softmax(b_ij, dim=2)  # Shape: [batch_size, in_capsules, num_capsules, 1]

            # Compute s_j: weighted sum over all u_hat
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)  # Shape: [batch_size, 1, num_capsules, out_dim]

            # Apply squash function
            v_j = self.squash(s_j)  # Shape: [batch_size, 1, num_capsules, out_dim]

            # Update b_ij
            if iteration < self.routing_iterations - 1:
                # Compute agreement a_ij = u_hat • v_j
                a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)  # Shape: [batch_size, in_capsules, num_capsules, 1]
                b_ij = b_ij + a_ij

        v_j = v_j.squeeze(1)  # Shape: [batch_size, num_capsules, out_dim]
        return v_j
    
    def squash(self, s_j, dim=-1):
        '''Squashes an input Tensor so it has a magnitude between 0 and 1.
           param s_j: input tensor to squash
           return: squashed tensor
        '''
        squared_norm = (s_j ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        s_j_sqrt = torch.sqrt(squared_norm + 1e-9)
        v_j = scale * s_j / s_j_sqrt
        return v_j