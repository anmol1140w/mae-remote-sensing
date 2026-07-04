import torch
import torch.nn as nn
from .encoder import Encoder

class ViTClassifier(nn.Module):
    """
    Vision Transformer for Classification (Fine-tuning MAE).
    """
    def __init__(
        self, 
        encoder: Encoder, 
        num_classes: int = 10, 
        global_pool: bool = False
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.global_pool = global_pool
        
        # Use encoder's embed_dim for the head
        embed_dim = encoder.cls_token.shape[-1]
        
        if global_pool:
            self.fc_norm = nn.LayerNorm(embed_dim)
        else:
            self.fc_norm = None
            
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize head
        torch.nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)
        
    def forward_features(self, x):
        x = self.encoder.patch_embed(x)
        x = x + self.encoder.pos_embed[:, 1:, :]

        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.encoder.blocks:
            x = blk(x)

        x = self.encoder.norm(x)

        return x[:, 0]   # CLS embedding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward (no masking during fine-tuning)
        # x, _, _ = self.encoder(x, mask_ratio=0.0)
        
        # Manually call encoder parts to avoid masking logic overhead if needed
        # but for simplicity, we use the forward method
        latent, _, _ = self.encoder(x, mask_ratio=0.0)
        
        if self.global_pool:
            # Global average pooling over all tokens (excluding cls token)
            x = latent[:, 1:, :].mean(dim=1)
            x = self.fc_norm(x)
        else:
            # Use only the cls token
            x = latent[:, 0]
            
        x = self.head(x)
        return x
