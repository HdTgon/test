"""LADA-style zero-shot reference and MoRAM CLIP; all routed LoRA lives here."""
import copy
import math
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from .clip.model import CLIP


class ZeroShotCLIP(nn.Module):
    """LADA zero-shot reference: normalize per prompt, average, normalize again.

    Keep the original CLIP encoders. Copy the supplied backbone so references
    and trainable models cannot mutate one another's weights or module tree.
    """
    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.clip_model = copy.deepcopy(clip_model)
        self.clip_model.requires_grad_(False)
        self.register_buffer('text_features', None)
        self.register_buffer('prompts', None)

    @property
    def image_encoder(self):
        return self.clip_model.visual

    @property
    def dtype(self):
        return self.clip_model.dtype

    @property
    def logit_scale(self):
        return self.clip_model.logit_scale.exp()

    def encode_text(self, text):
        return self.clip_model.encode_text(text)

    def encode_image(self, image):
        return self.clip_model.encode_image(image)

    @torch.no_grad()
    def init_text_features(self, classnames, templates, tokenize, device):
        if not isinstance(templates, list):
            templates = [templates]
        zeroshot_weights = []
        for classname in classnames:
            classname = classname.replace('_', ' ')
            texts = tokenize([template.format(classname) for template in templates]).to(device)
            class_embeddings = self.encode_text(texts)
            class_embeddings = F.normalize(class_embeddings, dim=-1)
            class_embedding = F.normalize(class_embeddings.mean(dim=0), dim=-1)
            zeroshot_weights.append(class_embedding)
        self.text_features = torch.stack(zeroshot_weights, dim=0)

    def forward(self, image):
        if self.text_features is None:
            raise RuntimeError('Call init_text_features before zero-shot classification')
        image_features = F.normalize(self.encode_image(image), dim=-1)
        return self.logit_scale * image_features @ self.text_features.t()


class MoRAM_CLIP(ZeroShotCLIP):
    """LADA CLIP with only MoRAM factors trainable; no DPT/LADA/Text_Tuner.

    forward(image) uses init_prompts or cached init_text_features;
    forward(image, text=tokens) computes a fresh textual classifier.
    All forward forms return a [batch, classes] logits tensor, like LADA.
    """
    def __init__(self, cfg, clip_model: CLIP):
        super().__init__(clip_model)
        apply_lora(self.clip_model, cfg)

    @torch.no_grad()
    def init_prompts(self, classnames, templates, tokenize, device):
        if isinstance(templates, str):
            templates = [templates]
        self.prompts = tokenize([templates[0].format(name.replace('_', ' '))
                                for name in classnames]).to(device)
        self.text_features = None

    def forward(self, image, text=None, text_features=None):
        if text_features is None:
            if text is not None:
                text_features = self.encode_text(text)
            elif self.prompts is not None:
                text_features = self.encode_text(self.prompts)
            elif self.text_features is not None:
                text_features = self.text_features
            else:
                raise RuntimeError('Provide text, text_features, or call init_prompts/init_text_features')
        image_features = F.normalize(self.encode_image(image), dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        return self.logit_scale * image_features @ text_features.t()


class LoRALayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights


class MoRAM_Linear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        module_name=None,
        lora_threshold: float = 1.0,
        temp: float = 0.01,
        topk: Optional[int] = None,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out

        self.module_name = module_name
        self.rank = r
        self.init_rank = r
        self.in_features = in_features
        self.out_features = out_features
        self.temp = float(temp)

        self.lora_threshold = float(lora_threshold)
        self.topk = int(topk) if topk is not None else r

        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.lora_scaling = 1
            self.weight.requires_grad = False
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

        self.old_lora_A = None
        self.old_lora_B = None
        self.old_rank = 0

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if getattr(self, 'lora_A', None) is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def set_old_rank(self, old_rank):
        self.old_rank = old_rank

    def increment_task(self):
        if self.lora_A is not None:
            if self.old_lora_A is not None:
                self.old_lora_A = nn.Parameter(torch.cat([
                    self.old_lora_A.data.clone(),
                    self.lora_A.data.clone(),
                ], dim=0))
                self.old_lora_B = nn.Parameter(torch.cat([
                    self.old_lora_B.data.clone(),
                    self.lora_B.data.clone(),
                ], dim=1))
            else:
                self.old_lora_A = nn.Parameter(self.lora_A.data.clone())
                self.old_lora_B = nn.Parameter(self.lora_B.data.clone())

            self.old_lora_A.requires_grad = False
            self.old_lora_B.requires_grad = False

            self.old_rank += self.rank

        if self.init_rank > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((self.init_rank, self.in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((self.out_features, self.init_rank)))
            self.register_buffer("gate", torch.zeros(1, self.init_rank, device=self.weight.device, dtype=self.weight.dtype))
            self.n_gate = 0
            self.lora_scaling = 1
            self.rank = self.init_rank
            self.r = self.init_rank
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.first_iter = True
            
    def forward(self, x: torch.Tensor, get_cur_feat=False, **kwargs):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)

        if self.old_lora_A is None and self.lora_A is None:
            return result, 0

        if self.old_lora_A is not None and self.lora_A is not None:
            combined_lora_A = torch.cat([self.old_lora_A, self.lora_A], dim=0)
            B_combined = torch.cat(
                [self.old_lora_B, self.lora_B * self.lora_scaling], dim=1
            )
        elif self.old_lora_A is not None:
            combined_lora_A = self.old_lora_A
            B_combined = self.old_lora_B
        elif self.lora_A is not None:
            combined_lora_A = self.lora_A
            B_combined = self.lora_B * self.lora_scaling
        else:
            return result, 0

        xd = self.lora_dropout(x)
        projection = xd @ combined_lora_A.T

        normed_projection = F.normalize(projection, p=2, dim=-1)

        k = min(max(self.topk, 2), normed_projection.shape[-1])
        topk_values, topk_indices = torch.topk(normed_projection, k=k, dim=-1)
        masked_logits = torch.full_like(normed_projection, float('-inf'))
        masked_logits.scatter_(-1, topk_indices, topk_values)
        weights = F.softmax(masked_logits / self.temp, dim=-1)

        if not self.training and self.lora_threshold > 0:
            gate_mask = (normed_projection >= self.lora_threshold).float()
            weights = weights * gate_mask

        weighted_proj = projection * weights
        lora_delta = torch.matmul(weighted_proj, B_combined.T)

        result = result + lora_delta

        loss = 0

        return result, loss



class PlainMultiheadAttentionLoRA(nn.Module):

    def __init__(self, existing_mha: nn.MultiheadAttention, module_name: str, enable_lora: list=['q', 'k', 'v', 'o'], r: int=0, lora_temp: float=0.01, lora_threshold: float=1.0, lora_topk: Optional[int]=None, coverage: Optional[float]=None, **kwargs):
        super(PlainMultiheadAttentionLoRA, self).__init__()
        self.module_name = module_name
        self.lora_temp = float(lora_temp)
        self.lora_threshold = float(lora_threshold)
        self.lora_topk = int(lora_topk) if lora_topk is not None else r
        coverage = 0.99 if coverage is None else float(coverage)
        self.router_coverage = min(max(coverage, 0.0), 1.0)
        self.embed_dim = existing_mha.embed_dim
        self.num_heads = existing_mha.num_heads
        self.batch_first = existing_mha.batch_first
        self.head_dim = self.embed_dim // self.num_heads
        self._qkv_same_embed_dim = existing_mha._qkv_same_embed_dim
        self.dropout = existing_mha.dropout
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.out_proj.bias is not None)
        self._init_from_existing_mha(existing_mha)
        self._apply_lora(enable_lora, r)
        self.to(device=existing_mha.out_proj.weight.device, dtype=existing_mha.out_proj.weight.dtype)
        self.scaled_dot_product_attention = F.scaled_dot_product_attention

    def _init_from_existing_mha(self, existing_mha):
        """
        Initializes the new attention module's weights with the existing MultiheadAttention weights.
        """
        existing_weight = existing_mha.in_proj_weight.data
        existing_bias = existing_mha.in_proj_bias.data if existing_mha.in_proj_bias is not None else None
        self.q_proj.weight.data.copy_(existing_weight[:self.embed_dim, :])
        if existing_bias is not None:
            self.q_proj.bias.data.copy_(existing_bias[:self.embed_dim])
        self.k_proj.weight.data.copy_(existing_weight[self.embed_dim:2 * self.embed_dim, :])
        if existing_bias is not None:
            self.k_proj.bias.data.copy_(existing_bias[self.embed_dim:2 * self.embed_dim])
        self.v_proj.weight.data.copy_(existing_weight[2 * self.embed_dim:, :])
        if existing_bias is not None:
            self.v_proj.bias.data.copy_(existing_bias[2 * self.embed_dim:])
        self.out_proj.weight.data.copy_(existing_mha.out_proj.weight.data)
        if self.out_proj.bias is not None:
            self.out_proj.bias.data.copy_(existing_mha.out_proj.bias.data)

    def _apply_lora(self, enable_lora, r):
        """
        Converts the specified projections into LoRA-enhanced projections.
        Ensure the pretrained weights are copied to the LoRA layers.
        """
        lora_kwargs = {'temp': self.lora_temp, 'lora_threshold': self.lora_threshold, 'topk': self.lora_topk}
        if 'q' in enable_lora:
            lora_q_proj = MoRAM_Linear(self.q_proj.in_features, self.q_proj.out_features, r=r, module_name=self.module_name + '.q_proj', **lora_kwargs)
            lora_q_proj.weight.data.copy_(self.q_proj.weight.data)
            if self.q_proj.bias is not None:
                lora_q_proj.bias.data.copy_(self.q_proj.bias.data)
            self.q_proj = lora_q_proj
        if 'k' in enable_lora:
            lora_k_proj = MoRAM_Linear(self.k_proj.in_features, self.k_proj.out_features, r=r, module_name=self.module_name + '.k_proj', **lora_kwargs)
            lora_k_proj.weight.data.copy_(self.k_proj.weight.data)
            if self.k_proj.bias is not None:
                lora_k_proj.bias.data.copy_(self.k_proj.bias.data)
            self.k_proj = lora_k_proj
        if 'v' in enable_lora:
            lora_v_proj = MoRAM_Linear(self.v_proj.in_features, self.v_proj.out_features, r=r, module_name=self.module_name + '.v_proj', **lora_kwargs)
            lora_v_proj.weight.data.copy_(self.v_proj.weight.data)
            if self.v_proj.bias is not None:
                lora_v_proj.bias.data.copy_(self.v_proj.bias.data)
            self.v_proj = lora_v_proj
        if 'o' in enable_lora:
            lora_out_proj = MoRAM_Linear(self.out_proj.in_features, self.out_proj.out_features, r=r, module_name=self.module_name + '.out_proj', **lora_kwargs)
            lora_out_proj.weight.data.copy_(self.out_proj.weight.data)
            if self.out_proj.bias is not None:
                lora_out_proj.bias.data.copy_(self.out_proj.bias.data)
            self.out_proj = lora_out_proj

    def forward(self, query, key, value, need_weights=False, key_padding_mask=None, attn_mask=None, is_causal=False, average_attn_weights=True, reg_prename=False, get_cur_feat=False):
        """
        Forward pass with LoRA-augmented attention layers.
        """
        if need_weights or key_padding_mask is not None:
            raise NotImplementedError("This adapter implements LADA CLIP attention (no attention weights or padding mask)")
        if query.ndim != 3:
            raise ValueError("LADA CLIP attention expects a 3D sequence")
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask = torch.zeros_like(attn_mask, dtype=query.dtype).masked_fill(attn_mask, float('-inf'))
            attn_mask = attn_mask.to(device=query.device, dtype=query.dtype)
        losses = []
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        tgt_len, bsz, embed_dim = query.shape
        source_len, _, _ = key.shape
        num_heads = self.num_heads
        if isinstance(embed_dim, torch.Tensor):
            head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, f'embed_dim {embed_dim} not divisible by num_heads {num_heads}'
        if isinstance(self.q_proj, MoRAM_Linear):
            q, prename_loss = self.q_proj(query, reg_prename=reg_prename, get_cur_feat=get_cur_feat)
            losses.append(prename_loss)
        else:
            q = self.q_proj(query)
        if isinstance(self.k_proj, MoRAM_Linear):
            k, prename_loss = self.k_proj(key, reg_prename=reg_prename, get_cur_feat=get_cur_feat)
            losses.append(prename_loss)
        else:
            k = self.k_proj(key)
        if isinstance(self.v_proj, MoRAM_Linear):
            v, prename_loss = self.v_proj(value, reg_prename=reg_prename, get_cur_feat=get_cur_feat)
            losses.append(prename_loss)
        else:
            v = self.v_proj(value)
        q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        source_len = k.size(1)
        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.dropout
        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, source_len, head_dim)
        v = v.view(bsz, num_heads, source_len, head_dim)
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        if isinstance(self.out_proj, MoRAM_Linear):
            attn_output, prename_loss = self.out_proj(attn_output, reg_prename=reg_prename, get_cur_feat=get_cur_feat)
            losses.append(prename_loss)
        else:
            attn_output = self.out_proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if not is_batched:
            attn_output = attn_output.squeeze(1)
        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(0, 1)
        return (attn_output, sum(losses) / len(losses) if len(losses) > 0 else 0)

    def train(self, mode: bool=True):
        """
        Overrides the train method to ensure LoRA-specific behavior is applied when training.
        """
        return super().train(mode)


class MoRAMLinearTensor(MoRAM_Linear):
    """Let LADA's original nn.Sequential MLP receive a tensor, not a tuple.

    The active upstream MoRAM implementation always returns regularization 0.
    """
    def forward(self, x):
        return super().forward(x)[0]


def mark_only_moram_trainable(model):
    for name, param in model.named_parameters():
        param.requires_grad_(name.endswith(('.lora_A', '.lora_B')))
    return model


def apply_lora(model, cfg):
    """Inject in-place into a loaded LADA CLIP; do not rebuild its encoders."""
    target = cfg.get('target_encoder', 'all')
    if target not in ('vision', 'text', 'all'):
        raise ValueError(f'Invalid target_encoder: {target}')
    targets = set(cfg.get('target_modules', ['q_proj', 'v_proj']))
    supported = {'q_proj', 'k_proj', 'v_proj', 'out_proj', 'ffn_in', 'ffn_out'}
    if not targets or targets - supported:
        raise ValueError(f'target_modules must be a nonempty subset of {sorted(supported)}')
    rank = int(cfg.get('rank', 4))
    if rank <= 0:
        raise ValueError('rank must be positive when injecting MoRAM')
    temp = float(cfg.get('temp', 0.01))
    if temp <= 0:
        raise ValueError('temp must be positive')
    threshold = float(cfg.get('prune_thre', 1.0))
    topk = int(cfg.get('topk', rank))
    transformers = []
    if target in ('vision', 'all'):
        if not hasattr(model.visual, 'transformer'):
            raise ValueError('Visual MoRAM injection requires a ViT backbone')
        transformers.append(('visual.', model.visual.transformer))
    if target in ('text', 'all'):
        transformers.append(('', model.transformer))
    # Reject repeated injection before mutating anything.
    if any(isinstance(m, (MoRAM_Linear, PlainMultiheadAttentionLoRA)) for m in model.modules()):
        raise ValueError('MoRAM is already injected')
    for prefix, transformer in transformers:
        for i, block in enumerate(transformer.resblocks):
            name = f'{prefix}transformer.resblocks.{i}'
            enabled = [letter for key, letter in [('q_proj','q'), ('k_proj','k'), ('v_proj','v'), ('out_proj','o')] if key in targets]
            if enabled:
                block.attn = PlainMultiheadAttentionLoRA(
                    block.attn, module_name=name+'.attn', enable_lora=enabled,
                    r=rank, lora_temp=temp, lora_threshold=threshold, lora_topk=topk)
            for key, attr in [('ffn_in','c_fc'), ('ffn_out','c_proj')]:
                if key not in targets:
                    continue
                original = getattr(block.mlp, attr)
                layer = MoRAMLinearTensor(original.in_features, original.out_features,
                    bias=original.bias is not None, r=rank, merge_weights=False,
                    module_name=name+'.mlp.'+attr, temp=temp,
                    lora_threshold=threshold, topk=topk,
                    device=original.weight.device, dtype=original.weight.dtype)
                with torch.no_grad():
                    layer.weight.copy_(original.weight)
                    if original.bias is not None:
                        layer.bias.copy_(original.bias)
                layer.train(original.training)
                setattr(block.mlp, attr, layer)
            block.attn.train(block.training)
    return mark_only_moram_trainable(model)


def increment_task(model):
    """Archive current factors, initialize new zero-delta factors, freeze history.

    Recreate the optimizer after this call: new Parameters replace old ones.
    """
    for module in model.modules():
        if isinstance(module, MoRAM_Linear):
            module.increment_task()
    return mark_only_moram_trainable(model)
