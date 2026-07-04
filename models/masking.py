import torch
from typing import Tuple

def random_masking(x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform random masking as per the MAE paper.
    
    Args:
        x (torch.Tensor): Input tokens of shape (B, L, D).
        mask_ratio (float): Ratio of patches to mask.
        
    Returns:
        x_masked (torch.Tensor): Masked tokens (only visible ones) of shape (B, L_visible, D).
        mask (torch.Tensor): Binary mask of shape (B, L), 0 for visible, 1 for masked.
        ids_restore (torch.Tensor): Indices to restore the original order of tokens.
    """
    B, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    
    # Generate random noise for each patch
    noise = torch.rand(B, L, device=x.device)  # (B, L)
    
    # Sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    # Keep the first len_keep indices
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    
    # Generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([B, L], device=x.device)
    mask[:, :len_keep] = 0
    # Unshuffle to get the binary mask in original order
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return x_masked, mask, ids_restore
