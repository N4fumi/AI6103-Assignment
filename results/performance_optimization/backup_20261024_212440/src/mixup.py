"""One Beta coefficient and one random pairing per minibatch."""
import numpy as np
import torch
import torch.nn.functional as F


def mixup_batch(x, y, alpha):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[permutation], y, y[permutation], lam


def mixup_loss(logits, a, b, lam):
    return lam * F.cross_entropy(logits, a) + (1 - lam) * F.cross_entropy(logits, b)
