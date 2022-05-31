import torch
import torch.nn as nn
import torch.nn.functional as F

ACT_MAPPING = {
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid
}