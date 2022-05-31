"""
Vocabulary and Tokenizer.
"""

from lib2to3.pgen2 import token
import os
import sys
import json
import numpy as np
from collections import OrderedDict, Counter
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import string


try:
    import jieba
except:
    None

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
DEFAULT_SPECIAL_TOKENS = ['<sos>', '<eos>', '<cls>', '<sep>']

class Vocabulary:
    """
    Vocabulary to map id to word, supporting build from scratch

    Args:
        word_vector: a dict mapping words to vectors or a np.array w.r.t. words
    """
    def __init__(self,
            words,
            special_tokens = DEFAULT_SPECIAL_TOKENS,
            word_vector: Optional[Union[np.array, dict]] = None
        ):
        self.special_tokens = [PAD_TOKEN, UNK_TOKEN] + special_tokens
        self.pad_idx = 0
        self.unk_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3
        vocab = {i: token for i,token in enumerate(self.special_tokens)}
        # exclude special tokens

        for i,token in enumerate(words):
            vocab[len(self.special_tokens) + i] = token
        
        assert len(list(vocab.keys())) == len(set(list(vocab.keys())))

        self.id2word = vocab
        self.word2id = {token:i for i,token in vocab.items()}

        if word_vector is not None:
            self.word_embeddings = self.get_word_embeddings(word_vector)
    
    def get_word_embeddings(self, word_vector):
        if isinstance(word_vector, dict):
            word_embs = np.array([
                word_vector[self.id2word[idx]] for idx in range(len(self.special_tokens), len(self.id2word))
            ])
        elif isinstance(word_vector, np.array):
            word_embs = word_vector
        else:
            raise TypeError(f"Unsupported type of word_vector: {type(word_vector)}")
        
        all_word_embs = np.zeros(
            (word_embs.shape[0] + len(self.special_tokens), word_embs.shape[1]),
            dtype = np.float32
        )
        all_word_embs[len(self.special_tokens):] = word_embs
        return all_word_embs
    
    def word_to_id(self, word):
        return self.word2id.get(word, self.unk_idx)
    
    def id_to_word(self, idx):
        return self.id2word[idx]
    
    def __len__(self):
        return len(self.id2word)

    @classmethod
    def from_scratch(cls, 
        text: List[str], 
        special_tokens = DEFAULT_SPECIAL_TOKENS, 
        size = None, 
        min_freq = None
    ):
        ct = Counter(text)
        
        min_freq = 1 if min_freq is None else 3
        words = [tk for tk, freq in ct.items() if freq >= min_freq]

        if size is not None:
            words = words[:size]
        return cls(words, special_tokens)
    """
    TODO: function of add or change special_tokens
    """
    def __str__(self):
        return (f'Vocabulary({self.id2word})')
    
    def load_word_vec_txt_line(self, path, skip = 1):
        print('Load word vectors')
        sk = 0
        mapping = {}
        with open(path) as f:
            for line in f:
                if sk < skip:
                    sk += 1
                    continue
                line = line.strip().split()
                v = np.array([float(k) for k in line[1:]])
                mapping[line[0]] = v
        return mapping
    
    def get_emb_mat(self, mapping):
        n_word = len(self.id2word)
        target = 0
        dim = list(mapping.values())[0].shape[0]
        mat = np.zeros((n_word, dim), dtype = np.float32)
        for wid, word in self.id2word.items():
            if word in mapping:
                mat[wid] = mapping[word]
                target += 1
        print(f'Target {target} / {n_word}')
        return mat

    def save_to_file(self, path):
        """
        Save words to file.
        """
        words = [self.id2word[i] for i in range(len(self.id2word))]
        with open(path, 'w') as f:
            f.write('\n'.join(words))

class Tokenizer:
    """
    Base class for all tokenizers.

    Default split by space.

    Args:
        pad: how to pad samples
            max_length: pad to max_length set in the args
            no: do not pad
        max_length (int): the padding max length.
    """

    def __init__(self, 
        vocab: Optional[Vocabulary] = None,
        pad = 'max_length', 
        max_length = None,
        lowercase = True
    ):
        self.vocab = vocab
        self.pad = pad
        self.max_length = max_length
        self.lowercase = lowercase

    def build_vocab(self, text_samples, size = None, min_freq = None) -> Vocabulary:
        """
        Build vocabulary from samples.
        """
        split_samples = [self.split_text(sample) for sample in text_samples]
        self.vocab = Vocabulary.from_scratch(
            [t for s in split_samples for t in s],
            size = size,
            min_freq = min_freq
        )
    
    def tokenize(self, text:str, return_np = True) -> List[Union[int, str]]:
        """
        Tokenize one sentence and return tokens or indexs and attention mask.

        Return:
            token_ids
            mask
        """
        if self.vocab is None:
            raise ValueError(f'Should build vocabulary before tokenization.')
        
        tokens = self.split_text(text)
        token_ids = [self.vocab.word_to_id(t) for t in tokens]
        
        if self.pad == 'max_length' and self.max_length is None:
            raise ValueError(f'No max_length for padding')
        
        if self.pad == 'max_length':
            token_ids, mask = self.pad_sample(token_ids)
        else:
            mask = [1] * len(token_ids)
        
        if return_np:
            return (
                np.array(token_ids, dtype = np.int64),
                np.array(mask, dtype = np.int64)
            )
        else:
            return token_ids, mask

    def pad_sample(self, token_ids):
        """
        Add sos and eos token and pad
        """
        max_len = self.max_length
        sent_len = min(len(token_ids), max_len - 2)
        pad_len = max(max_len - 2 - len(token_ids), 0)
        mask = [1] * (sent_len + 2) + [0] * pad_len
        token_ids = [self.vocab.sos_idx] + \
                token_ids[:sent_len] + \
                [self.vocab.eos_idx] + \
                [self.vocab.pad_idx] * pad_len
        return token_ids, mask

    def build_vocab_and_tokenize(self, text_samples, size = None, min_freq = None):
        """
        Build vocabulary of a corpus and tokenize it.

        Return:
            List[List[int]]
        """
        split_samples = [self.split_text(sample) for sample in text_samples]
        self.vocab = Vocabulary.from_scratch(
            [t for s in split_samples for t in s],
            size = size,
            min_freq = min_freq
        )
        tokenized_samples = [
            [self.vocab.word_to_id(t) for t in tokens] for tokens in split_samples
        ]

        if self.pad == 'max_length':
            if self.max_length is None:
                self.max_length = max([len(k) for k in split_samples])
            tokenized_samples_and_mask = [self.pad_sample(token_ids) for token_ids in tokenized_samples]
            tokenized_samples = [k[0] for k in tokenized_samples_and_mask]
            masks = [k[1] for k in tokenized_samples_and_mask]
        
        return tokenized_samples, masks

    def split_text(self, text: str) -> List[str]:
        """Split text into tokens"""
        return text.split(' ')

class JiebaWordTokenize(Tokenizer):
    def split_text(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        return list(jieba.cut(text))

class TokenMergeLetterTokenize(Tokenizer):
    def split_text(self, text:str) -> List[str]:
        split_text = []
        tmp = []
        if self.lowercase:
            text = text.lower()
        for c in text:
            if c in string.digits or c in string.ascii_letters:
                tmp.append(c)
            else:
                if len(tmp) > 0:
                    # add last word of characters
                    split_text.append(''.join(tmp))
                    tmp = []
                # add current word
                split_text.append(c)
        if len(tmp) > 0:
            split_text.append(''.join(tmp))
        return split_text

class CharacterTokenize(Tokenizer):
    def split_text(self, text:str) -> List[str]:
        return list(text)

if __name__ == '__main__':
    vocab = Vocabulary.from_scratch(['a', 'b', 'c', 'a', 'a'])
    print(vocab.word2id)

    tokenizer = TokenMergeLetterTokenize(vocab = None, max_length = 5)
    print(tokenizer.build_vocab_and_tokenize(['你好123.456我是John01']))
    print(tokenizer.vocab)