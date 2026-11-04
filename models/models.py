"""CLIP zero-shot inference and primal RAIL analytic incremental learning.

RAIL follows continual_clip_adaptor from:
https://github.com/linghan1997/Regression-based-Analytic-Incremental-Learning
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from clip.model import CLIP


class ZeroShotCLIP(nn.Module):
    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.clip_model = clip_model
        self.register_buffer('text_features', None)

    @property
    def image_encoder(self):
        return self.clip_model.visual

    @property
    def dtype(self):
        return self.clip_model.dtype

    @property
    def logit_scale(self):
        return self.clip_model.logit_scale.detach().exp()

    def encode_text(self, text):
        return self.clip_model.encode_text(text)

    def encode_image(self, image):
        return self.clip_model.encode_image(image.to(self.dtype))

    @torch.no_grad()
    def init_text_features(self, classnames, templates, tokenize, device):
        if isinstance(templates, str):
            templates = [templates]
        if not classnames or not templates:
            raise ValueError('classnames and templates must not be empty')
        weights = []
        for classname in classnames:
            texts = [template.format(classname.replace('_', ' ')) for template in templates]
            embeddings = F.normalize(self.encode_text(tokenize(texts).to(device)), dim=-1)
            weights.append(F.normalize(embeddings.mean(dim=0), dim=-1))
        self.text_features = torch.stack(weights)

    def forward(self, image):
        if self.text_features is None:
            raise RuntimeError('Call init_text_features before zero-shot inference')
        features = F.normalize(self.encode_image(image), dim=-1)
        return self.logit_scale * features @ self.text_features.t()


class RAILCLIP(ZeroShotCLIP):
    """Frozen CLIP with optional random expansion and recursive ridge regression.

    Accepts LADA's already-loaded CLIP instead of loading another backbone.
    cfg requires feature_expansion and regularization; hidden_dim is required
    only with expansion. analytic_adaption uses current_class_num (cumulative)
    and trained_class_num (previous total). Loader labels are task-local.
    Call init_text_features for zero-shot inference, analytic_adaption to fit,
    and forward for analytic logits. Preprocessing remains with the caller.
    """

    def __init__(self, cfg, clip_model: CLIP):
        super().__init__(clip_model)
        self.cfg = cfg
        self.feature_dim = clip_model.text_projection.shape[1]
        self.feature_expansion = bool(cfg.feature_expansion)
        self.hidden_dim = cfg.hidden_dim if self.feature_expansion else self.feature_dim
        if self.hidden_dim <= 0:
            raise ValueError('hidden_dim must be positive')
        device = clip_model.text_projection.device
        self.expansion_layer = (
            nn.Sequential(nn.Linear(self.feature_dim, self.hidden_dim, bias=False,
                                    device=device, dtype=torch.float32), nn.ReLU())
            if self.feature_expansion else nn.Identity()
        )
        self.analytic_adaptor = None
        self.register_buffer('R', None)
        self.clip_model.requires_grad_(False)
        self.expansion_layer.requires_grad_(False)
        self.clip_model.eval()

    def train(self, mode=True):
        super().train(mode)
        self.clip_model.eval()
        return self

    @torch.no_grad()
    def encode_images(self, images):
        return F.normalize(self.encode_image(images), dim=-1)

    def _analytic_features(self, images):
        # CLIP may use fp16; analytic operations use the expansion/state dtype.
        dtype = (self.expansion_layer[0].weight.dtype if self.feature_expansion
                 else self.R.dtype if self.R is not None else torch.float32)
        return self.expansion_layer(self.encode_images(images).to(dtype))

    @torch.no_grad()
    def analytic_adaption(self, task_id, cfg, train_loader, R=None):
        """Fit task 0 or update subsequent tasks; return the inverse covariance.

        R can be passed explicitly as in the reference; otherwise stored state
        is used. Labels are offset without mutating the loader's targets.
        """
        if task_id < 0:
            raise ValueError('task_id must be nonnegative')
        num_classes = int(cfg.current_class_num)
        previous = 0 if task_id == 0 else int(cfg.trained_class_num)
        if num_classes <= previous:
            raise ValueError('current_class_num must exceed trained_class_num')
        device = self.clip_model.text_projection.device
        dtype = (self.expansion_layer[0].weight.dtype if self.feature_expansion
                 else torch.float32)
        if task_id == 0:
            if cfg.regularization <= 0:
                raise ValueError('regularization must be positive')
            auto_cor = torch.zeros(self.hidden_dim, self.hidden_dim, device=device, dtype=dtype)
            crs_cor = torch.zeros(self.hidden_dim, num_classes, device=device, dtype=dtype)
        else:
            if self.analytic_adaptor is None or self.analytic_adaptor.out_features != previous:
                raise ValueError('Previous classifier does not match trained_class_num')
            R = self.R if R is None else R
            if R is None or R.shape != (self.hidden_dim, self.hidden_dim):
                raise ValueError('A matching inverse covariance R is required')
            R = R.to(device=device, dtype=dtype)
            weights = F.pad(self.analytic_adaptor.weight.t(), (0, num_classes - previous))

        self.clip_model.eval()
        sample_count = 0
        for images, targets in train_loader:
            targets = targets.to(device=device, dtype=torch.long) + previous
            if targets.numel() == 0:
                continue
            if targets.min() < previous or targets.max() >= num_classes:
                raise ValueError('Expected task-local labels in the current task class range')
            features = self._analytic_features(images.to(device))
            labels = F.one_hot(targets, num_classes).to(dtype)
            sample_count += features.shape[0]
            if task_id == 0:
                auto_cor += features.t() @ features
                crs_cor += features.t() @ labels
            else:
                # Woodbury update, followed by recursive least squares.
                cross = R @ features.t()
                system = torch.eye(features.shape[0], device=device, dtype=dtype) + features @ cross
                R = R - cross @ torch.linalg.solve(system, features @ R)
                weights = weights + R @ features.t() @ (labels - features @ weights)

        if sample_count == 0:
            raise ValueError('train_loader must contain training samples')
        if task_id == 0:
            system = auto_cor + cfg.regularization * torch.eye(self.hidden_dim, device=device, dtype=dtype)
            R = torch.linalg.solve(system, torch.eye(self.hidden_dim, device=device, dtype=dtype))
            weights = R @ crs_cor
        adaptor = nn.Linear(self.hidden_dim, num_classes, bias=False, device=device, dtype=dtype)
        adaptor.weight.copy_(weights.t())
        adaptor.requires_grad_(False)
        self.analytic_adaptor = adaptor
        self.R = R
        return R

    def forward(self, images):
        if self.analytic_adaptor is None:
            raise RuntimeError('Call analytic_adaption before analytic inference')
        return self.analytic_adaptor(self._analytic_features(images))

    def zero_shot(self, images, clip_weights=None):
        if clip_weights is None:
            if self.text_features is None:
                raise RuntimeError('Call init_text_features before zero-shot inference')
            clip_weights = self.text_features.t()
        # The reference RAIL fusion uses a fixed CLIP temperature of 100.
        return 100.0 * self.encode_images(images) @ clip_weights
