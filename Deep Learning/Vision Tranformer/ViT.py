import torch
import torch.nn as nn
import numpy as np

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # Shape: [batch_size, embed_dim, num_patches^(1/2), num_patches^(1/2)]
        x = x.flatten(2)  # Shape: [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # Shape: [batch_size, num_patches, embed_dim]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super(TransformerEncoder, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_ln1 = self.ln1(x)
        x_attn, _ = self.mha(x_ln1, x_ln1, x_ln1)
        x = x + x_attn
        x_ln2 = self.ln2(x)
        x_mlp = self.mlp(x_ln2)
        x = x + x_mlp
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, num_layers=12, num_heads=12, mlp_dim=3072, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)
        ])
        
        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)  # Shape: [batch_size, num_patches, embed_dim]
        batch_size = x.shape[0]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: [batch_size, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: [batch_size, num_patches + 1, embed_dim]
        x = x + self.pos_embed
        x = self.dropout(x)

        for encoder in self.transformer_encoders:
            x = encoder(x)

        x = self.ln(x)
        cls_token_final = x[:, 0]  # Shape: [batch_size, embed_dim]
        x = self.fc(cls_token_final)  # Shape: [batch_size, num_classes]
        return x

# Example usage:
img_size = 224
patch_size = 16
in_channels = 3
num_classes = 1000
embed_dim = 768
num_layers = 12
num_heads = 12
mlp_dim = 3072
dropout = 0.1

model = VisionTransformer(img_size, patch_size, in_channels, num_classes, embed_dim, num_layers, num_heads, mlp_dim, dropout)
x = torch.randn(1, 3, 224, 224)  # Example input: batch of 1 image, 3 channels, 224x224
logits = model(x)
print(logits.shape)  # Output shape: [1, 1000]