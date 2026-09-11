# models/clip_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder

class CLIPModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, pretrained_image=True):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim, pretrained_image)
        self.text_encoder = TextEncoder(vocab_size, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def forward(self, images, token_ids, attn_mask=None):
        img_emb = self.image_encoder(images)
        txt_emb = self.text_encoder(token_ids, attn_mask)
        scale = self.logit_scale.exp()
        return img_emb, txt_emb, scale

    def clip_loss(self, img_emb, txt_emb, scale):
        logits = scale * img_emb @ txt_emb.t()
        labels = torch.arange(img_emb.size(0), device=img_emb.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        return (loss_i + loss_t) / 2
