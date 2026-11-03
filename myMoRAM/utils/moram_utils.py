# Imported from Artificer-AI-Lab/MoRAM XTAIL/utils.py.
# LR functions are unchanged; LoRA I/O adds device/dtype safety and task metadata.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import os
import pickle
import random
import logging


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def torch_save_lora(model, save_path, lora_module=None, *, extra_state=None):
    """Save MoRAM factors/ranks, optionally with trainer prototype/task metadata."""
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    lora_params = ['lora_A', 'lora_B', 'lora_scaling', 'old_lora_A', 'old_lora_B',
                   'matrix', 'feature_list', 'feature_mat', 'gate', 'prototype']
    lora_state_dict = {k: v for k, v in model.state_dict().items()
                       if any(param_name in k for param_name in lora_params)}
    lora_ranks = {}
    if lora_module is not None:
        for name, module in model.named_modules():
            if isinstance(module, lora_module):
                lora_ranks[name] = dict(rank=module.rank, old_rank=module.old_rank,
                                        init_rank=module.init_rank,
                                        lora_scaling=module.lora_scaling)
    checkpoint = dict(state_dict=lora_state_dict, lora_ranks=lora_ranks)
    if extra_state:
        if set(extra_state) & set(checkpoint):
            raise ValueError('extra_state cannot override state_dict/lora_ranks')
        checkpoint.update(extra_state)
    torch.save(checkpoint, save_path)
    print('Checkpoint saved to', save_path)




def torch_load_lora(model, checkpoint_path, lora_module):
    """Restore ranks and factors on each module's existing device and dtype.

    Create the optimizer after loading, because Parameters are replaced.
    Base CLIP must already be loaded from the same original checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    lora_state_dict = checkpoint['state_dict']
    lora_ranks = checkpoint['lora_ranks']
    modules = {name: module for name, module in model.named_modules()
               if isinstance(module, lora_module)}
    if set(modules) != set(lora_ranks):
        raise ValueError('LoRA module paths differ from checkpoint; use the same injection config')
    for name, module in modules.items():
        info = lora_ranks[name]
        rank, old_rank = int(info['rank']), int(info['old_rank'])
        if rank < 0 or old_rank < 0:
            raise ValueError('Invalid saved LoRA ranks')
        module.rank = module.r = rank
        module.old_rank = old_rank
        module.init_rank = int(info.get('init_rank', module.init_rank))
        module.lora_scaling = info.get('lora_scaling', 1)
        for attribute, shape, trainable in [
            ('lora_A', (rank, module.in_features), True),
            ('lora_B', (module.out_features, rank), True),
            ('old_lora_A', (old_rank, module.in_features), False),
            ('old_lora_B', (module.out_features, old_rank), False),
        ]:
            size = old_rank if attribute.startswith('old_') else rank
            key = f'{name}.{attribute}'
            if size:
                if key not in lora_state_dict or tuple(lora_state_dict[key].shape) != shape:
                    raise ValueError(f'Invalid saved factor: {key}')
                setattr(module, attribute, nn.Parameter(module.weight.new_zeros(shape), requires_grad=trainable))
            else:
                setattr(module, attribute, None)
        # increment_task registers gate dynamically; it must exist before load.
        gate_key = f'{name}.gate'
        if gate_key in lora_state_dict:
            gate = lora_state_dict[gate_key].to(device=module.weight.device, dtype=module.weight.dtype)
            if 'gate' in module._buffers:
                module.gate = torch.zeros_like(gate)
            else:
                if hasattr(module, 'gate'):
                    delattr(module, 'gate')
                module.register_buffer('gate', torch.zeros_like(gate))
        elif 'gate' in module._buffers:
            delattr(module, 'gate')
    result = model.load_state_dict(lora_state_dict, strict=False)
    lora_params = ('lora_A', 'lora_B', 'lora_scaling', 'old_lora_A', 'old_lora_B',
                   'matrix', 'feature_list', 'feature_mat', 'gate', 'prototype')
    missing_lora = [key for key in result.missing_keys if any(p in key for p in lora_params)]
    if result.unexpected_keys or missing_lora:
        raise ValueError(f'Incomplete LoRA checkpoint: missing={missing_lora}, unexpected={result.unexpected_keys}')
    print(f'LoRA ranks and parameters have been successfully loaded from {checkpoint_path}.')
    return model


def torch_save(classifier, save_path):
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"state_dict": classifier.state_dict()}, save_path)
    print("Checkpoint saved to", save_path)

def torch_load(classifier, save_path, device=None):
    checkpoint = torch.load(save_path)
    try:
        missing_keys, unexpected_keys = classifier.load_state_dict(
            checkpoint["state_dict"], strict=False
        )
    except Exception:
        # added for loading LoRA parameters trained with CIL
        missing_keys, unexpected_keys = classifier.load_state_dict(
            checkpoint, strict=False
        )
    logging.info(f"Checkpoint loaded from {save_path}")

    if device is not None:
        classifier = classifier.to(device)
    return classifier

def torch_load_part(classifier, save_path, device=None):
    checkpoint = torch.load(save_path)
    missing_keys, unexpected_keys = classifier.load_state_dict(
        checkpoint["state_dict"], strict=False
    )
    logging.info(f"Checkpoint loaded from {save_path}")

    if device is not None:
        classifier = classifier.to(device)
    return classifier


def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class kernel_layer(nn.Module):
    def __init__(self, sv, gamma):
        super(kernel_layer, self).__init__()
        self.sv = sv
        self.gamma = gamma

    def forward(self, x):
        return kernel(x, self.sv, gamma=self.gamma)


def cls_acc(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def one_hot_cls_acc(output, target):
    if isinstance(output, np.ndarray) and isinstance(target, np.ndarray):
        pred = np.argmax(output, axis=1)
        labels = np.argmax(target, axis=1)
        correct_predictions = np.equal(pred, labels)
        acc = np.mean(correct_predictions.astype(float)) * 100
    elif torch.is_tensor(output) and torch.is_tensor(target):
        pred = torch.argmax(output, dim=1)
        labels = torch.argmax(target, dim=1)
        correct_predictions = torch.eq(pred, labels)
        acc = torch.mean(correct_predictions.float()) * 100
    else:
        raise ValueError('Unsupported types for prediction and target.')
    return acc


def kernel(x, X, gamma):
    """
    Args:
        x: input data
        X: static center embeddings
        gamma: Guassian kernel hyperparameter
    """
    with torch.no_grad():
        btch = 32
        ker = torch.exp(((X[:btch, :] - x.unsqueeze(1)) ** 2).sum(dim=-1).mul_(-1. * gamma))
        for i in range(1, math.ceil(X.size(0) / btch)):
            ker_new = torch.exp(
                ((X[i * btch:(i + 1) * btch, :] - x.unsqueeze(1)) ** 2).sum(dim=-1).mul_(-1. * gamma))
            ker = torch.cat((ker, ker_new), 1)
    return ker


def gaussian_kernel(x, X, gamma):
    distance = pairwise_distances(x, X, metric='euclidean', squared=True)
    return np.exp(-gamma * distance)


def linear_kernel(x, X):
    return x @ X.T


def cos_kernel(x, X):
    return 1 - linear_kernel(x, X)


def sample_per_class(dataset, n, num_classes=1000):
    indices_per_class = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(dataset.imgs):
        indices_per_class[label].append(idx)

    sampled_indices = [idx for indices in indices_per_class for idx in np.random.choice(indices, n, replace=False)]
    return sampled_indices


class kernel_ridge_regression:
    def __init__(self, lamda=0.1, gamma=0.1):
        self.lamda = lamda
        self.gamma = gamma
        self.alpha = None
        self.kernel = None

    def train(self, X, Y):
        """
        Gaussian kernel only
        """
        self.kernel = kernel(X, X, gamma=self.gamma).cpu().numpy()
        self.alpha = np.mat(self.kernel + self.lamda * np.eye(self.kernel.shape[0])).I @ Y
        return self.alpha

    def predict(self, X, X_train):
        """
        Args:
            X: on-device tensor
            X_train: on-device tensor
        Returns:
            on-cpu numpy
        """
        predictions = kernel(X, X_train, gamma=self.gamma).cpu().numpy() @ self.alpha
        return predictions
