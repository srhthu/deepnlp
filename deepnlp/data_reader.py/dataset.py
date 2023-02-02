import torch
from torch.utils.data import Dataset

from pathlib import Path

class CachedDataset(Dataset):
    """
    Load features from cache or make features from original data. Can save padded data to cache, or process the cached features for padding.

    Call init or __init__ in sub classes.

    Args:
        - cache_path: ``None``: no use of cache. ``(str)``: path of cache.

    Implement funcions for down classes:
        - make_features_from_origin: set the value of attr:features

    Attributes:
        - features: List[Dict[str, Any]]
    """
    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.init()
    
    def init(self):
        # make cache dir
        if self.cache_path is not None:
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok = True)
        
        # load features from cache
        self.features = self.load_from_cache()
        if not self.features:
            # read raw data and make features
            self.features = self.make_features_from_origin()
            self.save_to_cache()
        
        self.post_process()


    def load_from_cache(self):
        if self.cache_path and self.cache_path.exists():
            print(f'Load cached data: {self.cache_path}')
            return torch.load(self.cache_path)
        elif not self.cache_path:
            print('Cache is disabled')
            return None
    
    def save_to_cache(self):
        if self.cache_path is not None:
            torch.save(self.features, self.cache_path)

    def make_features_from_origin(self):
        """To be Customized."""
        raise NotImplementedError
    
    def post_process(self):
        """To be Customized"""
        return None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        fea = self.features[index]
        return fea