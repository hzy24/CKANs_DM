import torch
import torch.nn as nn
from kan import *



# A demo kanConv2d you can use "model.kanConv2d.fc.auto_symbolic()" to symbolic
class kanConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,device=device):
        super(kanConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.device = device
        self.fc = KAN(width=[in_channels * kernel_size * kernel_size, out_channels],device=device ,grid=3, k=3, seed=0).to(self.device)
    
    def forward(self, x):
        # Get batch size and input dimensions
        batch_size, _, height, width = x.size()
        
        # Calculate output dimensions
        output_height = height - self.kernel_size + 1
        output_width = width - self.kernel_size + 1
        
        # Create an empty output tensor
        output = torch.zeros(batch_size, self.out_channels, output_height, output_width, device=x.device)
        
        # Perform sliding window operation
        for i in range(output_height):
            for j in range(output_width):
                # Extract the patch
                patch = x[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                # Flatten the patch
                patch = patch.reshape(batch_size, -1)
                # Apply the linear layer
                output[:, :, i, j] = self.fc(patch)
        
        return output