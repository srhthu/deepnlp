"""Data reader for CAIL2018 legal judgment prediction dataset"""
from typing import Tuple, Dict, Any, List, Optional
import json
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer

from .dataset import CachedDataset

class CAIL_LJP_post(CachedDataset):
    """
    Features:
        - token_ids
        - mask
        - label_article
        - label_charge
        - label_penalty
    """
    def __init__(self, data_path, cache_path: Optional[str],
                 tokenizer: Optional[PreTrainedTokenizer],
                 max_length: int = 512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        super().__init__(cache_path)
    
    def make_features_from_origin(self):
        print(f'Read original data: {self.data_path}')
        with open(self.data_path) as f:
            data = [json.loads(k) for k in f]
        
        features = []
        for sample in tqdm(data, ncols=80):
            encodings = self.tokenizer(sample['text'], padding = 'max_length', 
                           truncation = True, max_length = self.max_length)
            fea = {
                'token_ids': encodings.input_ids,
                'mask': encodings.attention_mask,
                'label_article': sample['label_article'],
                'label_charge': sample['label_charge'],
                'label_penalty': sample['label_penalty']
            }
            features.append(fea)
        return features