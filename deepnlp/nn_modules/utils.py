"""
Utilities
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def sequencial_pool(x: torch.Tensor, pool_type: str):
    """
    Pool on x (batch, seq_len, hidden)
    """
    if pool_type == 'max':
        return torch.max(x, dim = 1)[0]
    elif pool_type == 'mean':
        return torch.mean(x, dim = 1)
    else:
        raise ValueError(f'unknown pool: {pool_type}')