"""Example-weighted means; all accuracy values are fractions, not percentages."""
import math
import torch


def weighted_correct(logits, a, b=None, lam=1.0):
    predictions = logits.argmax(dim=1)
    if b is None:
        b = a
    return float((lam * (predictions == a).float() + (1 - lam) * (predictions == b).float()).sum().item())


class Metrics:
    def __init__(self):
        self.loss_sum = self.correct = 0.0
        self.n = 0

    def update(self, loss, correct, n):
        if not math.isfinite(loss) or not math.isfinite(correct):
            raise FloatingPointError('Nonfinite metric. Use a fresh FP32 run; do not change precision mid-experiment.')
        self.loss_sum += float(loss) * n
        self.correct += correct
        self.n += n

    def values(self):
        if not self.n:
            raise ValueError('Empty data loader')
        return self.loss_sum / self.n, self.correct / self.n


def validation_key(row):
    # Maximize accuracy, then minimize CE loss; earliest epoch wins exact ties.
    return (-row['val_accuracy'], row['val_loss'], row['epoch'])
