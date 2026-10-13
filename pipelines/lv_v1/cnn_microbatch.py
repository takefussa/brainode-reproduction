"""Accumulate soft BCE gradients, preserving the effective optimizer batch.

The cognition CNN has no batch normalization or cross-sample operations.
Each example contributes 1/N of its BCE gradient, then AdamW steps once.
"""
import torch
from torch.nn import functional as F


def backward_soft_bce_batch(model, x, y, microbatch_size):
    if len(x)!=len(y) or not len(x) or microbatch_size<1:
        raise ValueError('nonempty matching batches and positive microbatch required')
    total=real_total=pseudo_total=0.
    real_count=pseudo_count=0
    for start in range(0,len(x),microbatch_size):
        target=y[start:start+microbatch_size]
        losses=F.binary_cross_entropy_with_logits(model.logits(x[start:start+microbatch_size]),target,reduction='none')
        if not torch.isfinite(losses).all():raise ValueError('nonfinite CNN training loss')
        (losses.sum()/len(x)).backward()
        real=(target==0)|(target==1)
        total+=float(losses.detach().sum())
        real_total+=float(losses[real].detach().sum())
        pseudo_total+=float(losses[~real].detach().sum())
        real_count+=int(real.sum());pseudo_count+=int((~real).sum())
    return total,real_total,pseudo_total,real_count,pseudo_count
