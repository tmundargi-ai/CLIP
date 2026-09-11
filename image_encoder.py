# models/image_encoder.py
import torch
import torch.nn as nn
from torchvision.models import vit_b_32

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512, pretrained=False):
        super().__init__()
        vit = vit_b_32(weights="IMAGENET1K_V1" if pretrained else None)
        vit.heads = nn.Identity()  # remove classification head
        self.backbone = vit
        self.proj = nn.Linear(768, embed_dim)

    def forward(self, images):
        feats = self.backbone(images)          # (B, 768)
        feats = self.proj(feats)               # (B, embed_dim)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats
