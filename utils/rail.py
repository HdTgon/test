"""Gaussian kernel ridge regression used by reference dual_RAIL.py/utils.py.

https://github.com/linghan1997/Regression-based-Analytic-Incremental-Learning
"""
import torch


@torch.no_grad()
def gaussian_kernel(x, centers, gamma, chunk_size=32):
    """exp(-gamma * squared distance), chunking reference feature columns."""
    if gamma <= 0:
        raise ValueError('gamma must be positive')
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    blocks = []
    for start in range(0, centers.shape[0], chunk_size):
        distances = (x.unsqueeze(1) - centers[start:start + chunk_size]).square().sum(dim=-1)
        blocks.append(torch.exp(-gamma * distances))
    if not blocks:
        return x.new_empty((x.shape[0], 0))
    return torch.cat(blocks, dim=1)


@torch.no_grad()
def fit_kernel_ridge(features, labels, regularization, gamma):
    if regularization <= 0:
        raise ValueError('dual_regularization must be positive')
    # The reference uses NumPy's float64 solve after computing a float32
    # Gaussian kernel. Keep that precision, without np.mat or explicit inverse.
    kernel = gaussian_kernel(features, features, gamma).double()
    kernel.diagonal().add_(regularization)
    return torch.linalg.solve(kernel, labels.double())
