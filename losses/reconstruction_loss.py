import torch
import torch.nn as nn

class MAELoss(nn.Module):
    """
    MSE Loss for MAE reconstruction with optional pixel normalization.
    """
    def __init__(self, norm_pix_loss: bool = True):
        super().__init__()
        self.norm_pix_loss = norm_pix_loss

    def forward(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor, patch_size: int) -> torch.Tensor:
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs, patch_size)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def patchify(self, imgs: torch.Tensor, p: int) -> torch.Tensor:
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 * 3)
        """
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
