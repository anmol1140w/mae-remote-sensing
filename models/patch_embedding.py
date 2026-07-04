import torch
import torch.nn as nn
from typing import Tuple

class PatchEmbedding(nn.Module):
    """
    Convert image to patch embeddings.
    
    Args:
        img_size (int): Size of input image (assumed square).
        patch_size (int): Size of each patch (assumed square).
        in_chans (int): Number of input channels.
        embed_dim (int): Embedding dimension.
    """
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 16, 
        in_chans: int = 3, 
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        self.proj = nn.Conv2d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Patch embeddings of shape (B, L, D) where L is num_patches.
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}x{W}) doesn't match model image size ({self.img_size}x{self.img_size})"
        
        # (B, C, H, W) -> (B, D, H/P, W/P)
        x = self.proj(x)
        # (B, D, H/P, W/P) -> (B, D, L) -> (B, L, D)
        x = x.flatten(2).transpose(1, 2)
        return x
