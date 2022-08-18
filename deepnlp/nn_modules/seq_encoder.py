"""
Implementation of various sequence (ids) to vector models.
"""
import os
from turtle import forward
import numpy as np
import math
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig
from transformers.models.bert import BertModel, BertConfig

from .const import ACT_MAPPING

class SeqEncoder(nn.Module):
    """
    Base class of sequence (ids or vectors) to sequence encoder models.
    """
    def __init__(self):
        super().__init__()
    
    def get_output_dim(self):
        raise NotImplementedError()
    

class SeqEncoder_RNN(SeqEncoder):
    def __init__(self,
        rnn_type: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.,
        bidirectional: bool = True
    ):
        super().__init__()

        rnn_cls = nn.LSTM if rnn_type.lower() == 'lstm' else \
                  nn.GRU if rnn_type.lower() == 'gru' else None
        if not rnn_cls:
            raise ValueError(f'Unknown rnn_type: {rnn_type}')
        
        self.rnn = rnn_cls(input_size, hidden_size, num_layers, batch_first = True,
                            bidirectional = bidirectional, dropout = dropout
        )
    def forward(self, x):
        output, _ = self.rnn(x)
        return output
    
    def get_output_dim(self):
        return self.rnn.hidden_size*2 if self.rnn.bidirectional else self.rnn.hidden_size

class SeqEncoder_CNNRNN(SeqEncoder):
    """
    CNN + RNN for seq2seq encoder
    """
    def __init__(self,
        rnn_type: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        cnn_out_size: Optional[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.,
        bidirectional: bool = True,
        activation: str = 'tanh'
    ):
        super().__init__()

        if cnn_out_size is None:
            cnn_out_size = hidden_size
        self.cnn = nn.Conv1d(
            input_size,
            cnn_out_size,
            kernel_size = kernel_size
        )
        self.cnn_act = ACT_MAPPING[activation]()

        self.rnn = SeqEncoder_RNN(rnn_type, cnn_out_size, hidden_size, num_layers,
                            dropout=dropout, bidirectional=bidirectional
        )
    def forward(self, x):
        cnn_out = self.cnn(torch.transpose(x, 2, 1))
        # (batch_size, hid_dim, seq_len)
        cnn_out = torch.transpose(cnn_out, 2, 1)
        cnn_out = self.cnn_act(cnn_out)

        rnn_out = self.rnn(cnn_out)
        return rnn_out
    
    def get_output_dim(self):
        return self.rnn.get_output_dim()