"""LADA CLIP with optional routed MoRAM LoRA and no replacement CLIP classes."""
import logging
import torch
from .clip import clip
from .injection import apply_lora, increment_task, mark_only_moram_trainable
from .moram_lora import MoRAM_Linear


def get_moram(cfg, zero_shot=False):
    """Return (LADA CLIP, train_transform, test_transform, tokenizer).

    zero_shot=True: original frozen CLIP; False: only current MoRAM factors train.
    Both retain LADA's tensor encoders and (image_logits, text_logits) forward.
    """
    name = cfg.get('backbone_type', 'ViT-B/16').removeprefix('CLIP-')
    name = {'ViT-B-16':'ViT-B/16', 'ViT-B-32':'ViT-B/32',
            'ViT-L-14':'ViT-L/14', 'ViT-L-14-336':'ViT-L/14@336px'}.get(name, name)
    if cfg.get('pretrained_weight', 'openai') != 'openai':
        raise ValueError('Use OpenAI CLIP weights; checkpoint_path can specify a local OpenAI checkpoint')
    prec = cfg.get('prec', 'fp32')
    if prec not in ('fp16','fp32','amp'):
        raise ValueError(f'Invalid precision: {prec}')
    model, preprocess = clip.load(cfg.get('checkpoint_path',name), device='cpu', jit=False)
    if prec == 'fp16':
        from .clip.model import convert_weights
        convert_weights(model)
    else:
        model.float()
    if zero_shot:
        model.requires_grad_(False)
    else:
        apply_lora(model, cfg)
    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    logging.info('LADA CLIP + MoRAM: backbone=%s precision=%s zero_shot=%s',name,prec,zero_shot)
    return model, preprocess, preprocess, clip.tokenize


__all__ = ['get_moram', 'apply_lora', 'increment_task', 'MoRAM_Linear', 'clip', 'mark_only_moram_trainable']
