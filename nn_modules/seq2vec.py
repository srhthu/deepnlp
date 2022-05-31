"""
Implementation of various sequence (vectors) to vector models.
"""
import os
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
from .utils import sequencial_pool

# ------------
# Archetype
# ------------
class Seq2Vec(nn.Module):
    """
    Base class of sequence (vectors) to vector models.

    input:
        x: (batch, seq_len, hidden_size)
    output:
        h: (batch, hidden_size)
    """
    def __init__(self):
        super().__init__()
    
    def get_output_dim(self):
        raise NotImplementedError()
    
    @staticmethod
    def from_name(name):
        name = name.lower()
        if name == 'seq2vec_rnn':
            return Seq2Vec_RNN
        elif name == 'seq2vec_cnnrnn':
            return Seq2Vec_CNNRNN
        else:
            raise ValueError(f'Unknown model type: {name}')

class SentEncoder(nn.Module):
    """
    Base class for sent (seq of ids) to vector models.

    Input: (x, ...)
        x: (batch, seq_len)
        other args such as attention_mask for bert.
    Output:
        h: (batch, hidden_size)
    """
    def __init__(self):
        super().__init__()

    def get_output_dim(self):
        raise NotImplementedError()
    
    @staticmethod
    def from_name(name):
        name = name.lower()
        if name == 'sentencoder_emb':
            return SentEncoder_Emb
        elif name == 'sentencoder_bert':
            return SentEncoder_Bert
        else:
            raise ValueError(f'Unknown model type: {name}')



# ------------
# Seq2Vec models
# ------------
class Seq2Vec_RNN(Seq2Vec):
    """
    RNN encoder for seq2vec.
    
    Args
        rnn_type: [lstm, gru]
        pool_type: [last, mean, max]
    
    Input:
        x: (batch, seq_len, dim)
    """
    def __init__(self,
        rnn_type: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        pool_type: str,
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

        self.pool_type = pool_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.rnn.flatten_parameters() # fix data parallel warning

        output, (h_n, _) = self.rnn(x)
        pooled_out = self.pool(output, h_n)
        
        return pooled_out
    
    def pool(self, rnn_output, h_n):
        if self.pool_type == 'last':
            if self.rnn.bidirectional:
                fw = h_n[-2]
                bw = h_n[-1]
                # for output, it is [fw, bw]
                pooled_out = torch.cat((fw, bw), dim = -1)
            else:
                pooled_out = h_n[-1]
        else:
            pooled_out = sequencial_pool(rnn_output, self.pool_type)
        return pooled_out

    def get_output_dim(self):
        return self.rnn.hidden_size*2 if self.rnn.bidirectional else self.rnn.hidden_size

class Seq2Vec_CNNRNN(Seq2Vec_RNN):
    """
    CNN + RNN sequence to vector. Subclass of Seq2Vec_RNN
    """
    def __init__(self,
        rnn_type: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        pool_type: str,
        kernel_size: int = 3,
        dropout: float = 0.,
        bidirectional: bool = True,
        activation: str = 'tanh'
    ):
        super().__init__(
            rnn_type, hidden_size, hidden_size, num_layers,
            pool_type, dropout = dropout, bidirectional= bidirectional
        )

        self.cnn = nn.Conv1d(
            input_size,
            hidden_size,
            kernel_size = kernel_size
        )
        self.cnn_act = ACT_MAPPING[activation]()

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        self.rnn.flatten_parameters()

        cnn_out = self.cnn(torch.transpose(x, 2, 1))
        # (batch_size, hid_dim, seq_len)
        cnn_out = torch.transpose(cnn_out, 2, 1)
        cnn_out = self.cnn_act(cnn_out)

        output, (h_n, _) = self.rnn(cnn_out)
        pooled_out = self.pool(output, h_n)

        return pooled_out


# ------------
# SentEncoder models
# ------------
class SentEncoder_Emb(SentEncoder):
    """
    RNN for sent to vector.
    """
    def __init__(self,
        vocab_size: int,
        emb_dim: int,
        seq2vec_name: str,
        rnn_type: str,
        hidden_size: int,
        num_layers: int,
        pool_type: str,
        emb_dropout: 0.3,
        dropout: float = 0.,
        bidirectional: bool = True
    ):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.seq2vec = Seq2Vec.from_name(seq2vec_name)(
            rnn_type, emb_dim, hidden_size, num_layers, pool_type, 
            dropout = dropout, bidirectional = bidirectional
        )
    
    def forward(self, x, *args):
        h = self.emb_drop(self.emb(x))
        return self.seq2vec(h)
    
    def get_output_dim(self):
        return self.seq2vec.get_output_dim()


class SentEncoder_Bert(SentEncoder):
    """
    Bert encoder for Seq2Vec.

    Load pretrained parameters from bert_path and support customized number of layers.

    Args:
        pool_type: [first_token, max, mean]
    """
    def __init__(self,
        bert_path: str,
        num_layers: int,
        pool_type: str,
        bert_config: Optional[PretrainedConfig] = None
    ):
        super().__init__()

        bert_pretrain = BertModel.from_pretrained(bert_path)
        if bert_config is None:
            cfg_dt, cfg_kws = BertConfig.get_config_dict(bert_path)
            cfg_dt['num_hidden_layers'] = num_layers
            bert_config = BertConfig(**cfg_dt, **cfg_kws)
        
        self.bert = BertModel(bert_config)
        self.bert.load_state_dict(bert_pretrain.state_dict(), strict = False)        

        self.pool_type = pool_type

    def forward(self, input_ids, attention_mask = None, token_type_ids = None):
        bert_out = self.bert(input_ids, attention_mask, token_type_ids)
        if self.pool_type == 'first_token':
            pool_out = bert_out.pooler_output
        else:
            pool_out = sequencial_pool(bert_out.last_hidden_state, self.pool_type)
        
        return pool_out
    
    def get_output_dim(self):
        return self.bert.config.hidden_size