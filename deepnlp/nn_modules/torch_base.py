import os
import numpy as np
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Multi layer perceptron.

    Args:
        hidden_sizes: input and output dim from 1st to last layer.
    """
    def __init__(
        self,
        hidden_sizes: List[int],
        activation_cls: nn.Module = nn.Tanh,
        last_activation = True
    ):
        super().__init__()

        in_sizes = hidden_sizes[:-1]
        out_sizes = hidden_sizes[1:]
        mod_list = []
        for in_dim, out_dim in zip(in_sizes, out_sizes):
            mod_list.append(nn.Linear(in_dim, out_dim))
            mod_list.append(activation_cls())
        
        # for classification, last layer do not need activation.
        if not last_activation:
            mod_list.pop()        

        self.mlp = nn.Sequential(*mod_list)
    
    def forward(self, x: torch.Tensor):
        return self.mlp(x)


