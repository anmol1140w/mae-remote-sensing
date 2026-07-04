import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder (MAE) implementation.
    """
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 16, 
        in_chans: int = 3, 
        embed_dim: int = 1024, 
        depth: int = 24, 
        num_heads: int = 16, 
        decoder_embed_dim: int = 512, 
        decoder_depth: int = 8, 
        decoder_num_heads: int = 16, 
        mlp_ratio: float = 4.0, 
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        
        self.encoder = Encoder(
            img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, norm_layer
        )
        
        self.decoder = Decoder(
            self.encoder.patch_embed.num_patches, patch_size, in_chans, 
            embed_dim, decoder_embed_dim, decoder_depth, decoder_num_heads, mlp_ratio, norm_layer
        )

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 * 3)
        """
        p = self.encoder.patch_embed.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, L, patch_size**2 * 3)
        imgs: (N, 3, H, W)
        """
        p = self.encoder.patch_embed.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75):
        latent, mask, ids_restore = self.encoder(imgs, mask_ratio)
        pred = self.decoder(latent, ids_restore)
        return pred, mask
