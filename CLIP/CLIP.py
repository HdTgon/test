"""Plain CLIP ViT adapted from HotanLee/DeFT (no tuner or PEFT)."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .text_encoder import TextEncoder

class CLIP_ViT(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.proj = clip_model.visual.proj

    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def forward(self, x):
        x = x.to(self.dtype)
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        _bsz = x.shape[0]
        _emb_dim = x.shape[2]
        n_layers = self.transformer.layers
        for i in range(n_layers):
            block = self.transformer.resblocks[i]
            _seq_len = x.shape[1]
            x = x.permute(1, 0, 2)
            _attn = block.attn
            _ln_1 = block.ln_1
            _mlp = block.mlp
            _ln_2 = block.ln_2
            _attn_in_proj_weight = _attn.in_proj_weight
            _attn_in_proj_bias = _attn.in_proj_bias
            _attn_out_proj_weight = _attn.out_proj.weight
            _attn_out_proj_bias = _attn.out_proj.bias
            _mlp_in_proj = _mlp[0]
            _mlp_gelu = _mlp[1]
            _mlp_out_proj = _mlp[2]
            _num_heads = _attn.num_heads
            _head_dim = _emb_dim // _num_heads
            residual = x
            x = _ln_1(x)
            qkv = F.linear(x, _attn_in_proj_weight, _attn_in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.contiguous().view(q.shape[0], q.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            k = k.contiguous().view(k.shape[0], k.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            v = v.contiguous().view(v.shape[0], v.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            q = q / math.sqrt(_head_dim)
            attn = torch.bmm(q, k.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)
            x = torch.bmm(attn, v)
            x = x.transpose(0, 1).contiguous().view(-1, _emb_dim)
            x = F.linear(x, _attn_out_proj_weight, _attn_out_proj_bias)
            x = x.view(_seq_len, _bsz, _emb_dim)
            x = residual + x
            residual = x
            x = _ln_2(x)
            x = _mlp_in_proj(x)
            x = _mlp_gelu(x)
            x = _mlp_out_proj(x)
            x = residual + x
            x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        return x

class CLIP(nn.Module):
    """Wrap an OpenAI-format ViT backbone using DeFT's untuned encoders.

    encode_image/encode_text return tensors. forward returns CLIP logits.
    No freezing, CUDA move, learned prompts or adapter injection is implicit.
    """
    def __init__(self, clip_model):
        super().__init__()
        if not hasattr(clip_model.visual, "transformer"):
            raise ValueError("This DeFT visual encoder supports ViT CLIP backbones only")
        self.image_encoder = CLIP_ViT(clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.context_length = clip_model.context_length
        self.vocab_size = clip_model.vocab_size

    @property
    def dtype(self):
        return self.image_encoder.dtype

    def encode_image(self, image, normalize=False):
        features = self.image_encoder(image) @ self.image_encoder.proj
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize=False):
        features = self.text_encoder(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        logits_per_image = self.logit_scale.exp() * image_features @ text_features.t()
        return logits_per_image, logits_per_image.t()


Model = CLIP
