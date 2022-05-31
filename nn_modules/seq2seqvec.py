"""
Implementation of sequence (ids or vectors) encoder (to seq of vectors) models.
"""
"""
Reflection
seq2seqvec - rnn
seq2vec - rnn

sent2vec - rnn or bert
sent2seqvec - rnn or bert
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.models.bert import BertModel, BertConfig

from .const import ACT_MAPPING

class SeqPool(nn.Module):
    """
    Pool the output of a Seq2SeqVec model.
    """
    def __init__(self, pool_type: str):
        super().__init__()
        self.pool_type = pool_type.lower()
    
    def forward(self, x):
        # x: (batch, seq_len, dim)
        if self.pool_type == 'mean':
            return torch.mean(x, dim = 1)
        elif self.pool_type == 'max':
            return torch.max(x, dim = 1)[0]
        elif self.pool_type == 'last':
            return x[:,-1,:]
        elif self.pool_type == 'first':
            return x[:, 0, :]
        elif self.pool_type == 'last_bi':
            # pool the output of a bidirectional rnn
            # lstm output is concat of [fw, bw]
            dim = x.shape[2]
            fw = x[:, -1, :dim//2]
            bw = x[:, 0, dim//2:]
            return torch.cat([fw,bw], dim = -1)

# ------------
# Archetype
# ------------

class Seq2SeqVec(nn.Module):
    """
    Base class of sequence (vectors) to sequence (vectors) models.

    Input:
        x: (batch, seq_len, dim)
    Output:
        h: (batch, seq_len, dim)
    """
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def from_name(name):
        name = name.lower()
        if name == 'seq2seqvec_rnn':
            return Seq2SeqVec_RNN
        elif name == 'seq2seqvec_cnnrnn':
            return Seq2SeqVec_CNNRNN
        else:
            raise ValueError(f'Unknown model type: {name}')

class Sent2SeqVec(nn.Module):
    """
    Base class of sent ids to sequence (vectors) models.

    Input:
        x: (batch, seq_len)
    Output:
        h: (batch, seq_len, dim)
    """
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def from_name(name):
        name = name.lower()
        if name == 'sent2seqvec_emb':
            return Sent2SeqVec_Emb
        elif name == 'sent2seqvec_bert':
            return Sent2SeqVec_Bert
        else:
            raise ValueError(f'Unknown model type: {name}')

# ------------
# Seq2SeqVec models
# ------------
class Seq2SeqVec_RNN(Seq2SeqVec):
    def __init__(self,
        rnn_type: str,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
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
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        self.rnn.flatten_parameters()

        output, _ = self.rnn(x)
        return output
    
    def get_output_dim(self):
        return self.rnn.hidden_size*(1 + int(self.rnn.bidirectional))

class Seq2SeqVec_CNN(Seq2SeqVec):
    def __init__(self,
        input_size,
        hidden_size,
        kernel_size = 3,
        activation: str = 'tanh'
    ):
        super().__init__()

        if kernel_size % 2 != 1:
            raise RuntimeError(f'For Seq2SeqVec model, kernel_size ({kernel_size}) should be odd so that output has same length as input')
        
        self.cnn = nn.Conv1d(input_size, hidden_size, kernel_size,
                            padding = (kernel_size - 1) // 2)
        self.act = ACT_MAPPING[activation]()
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        cnn_out = self.cnn(torch.transpose(x, 2, 1))
        # (batch_size, hid_dim, seq_len)
        cnn_out = torch.transpose(cnn_out, 2, 1)
        cnn_out = self.act(cnn_out)
        return cnn_out


class Seq2SeqVec_CNNRNN(Seq2SeqVec_RNN):
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
        if cnn_out_size is None:
            cnn_out_size = hidden_size
        
        super().__init__(
            rnn_type, cnn_out_size, hidden_size, num_layers,
            dropout = dropout, bidirectional= bidirectional
        )
        
        if kernel_size % 2 != 1:
            raise RuntimeError(f'For Seq2SeqVec model, kernel_size ({kernel_size}) should be odd so that output has same length as input')
        self.cnn = nn.Conv1d(
            input_size,
            cnn_out_size,
            kernel_size = kernel_size,
            padding = (kernel_size - 1) // 2
        )
        self.cnn_act = ACT_MAPPING[activation]()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.rnn.flatten_parameters()

        cnn_out = self.cnn(torch.transpose(x, 2, 1))
        # (batch_size, hid_dim, seq_len)
        cnn_out = torch.transpose(cnn_out, 2, 1)
        cnn_out = self.cnn_act(cnn_out)

        output, (h_n, _) = self.rnn(cnn_out)

        return output


# ------------
# Sent2SeqVec models
# ------------
class Sent2SeqVec_Emb(Sent2SeqVec):
    """
    Embedding based Sent2SeqVec model
    forward(x)
    """
    def __init__(self,
        vocab_size: int,
        emb_dim: int,
        seq2seqvec_name: str,
        seq2seqvec_args: dict
    ):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, emb_dim)
        seq2seqvec_args['input_size'] = emb_dim
        self.seq2seqvec = Seq2SeqVec.from_name(seq2seqvec_name)(**seq2seqvec_args)
    
    def forward(self, x, *args, **kwargs):
        # to be compatible with bert
        return self.seq2seqvec(self.emb(x))
    
    def get_output_dim(self):
        return self.seq2seqvec.get_output_dim()

class Sent2SeqVec_Bert(Sent2SeqVec):
    """
    Bert encoder.

    Load pretrained parameters from bert_path and support customized number of layers.

    forward(input_ids, attention_mask, token_type_ids)
    """
    def __init__(self,
        bert_path: str,
        num_layers: int,
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

    def forward(self, input_ids, attention_mask = None, token_type_ids = None):
        bert_out = self.bert(input_ids, attention_mask, token_type_ids)
        
        return bert_out
    
    def get_output_dim(self):
        return self.bert.config.hidden_size 
