# %%
from huggingface_hub import (
    PyTorchModelHubMixin, hf_hub_download, scan_cache_dir,
    HFCacheInfo, snapshot_download
)
import os
from pathlib import Path
# %%
hf_hub_download('gpt2', filename='config.json')
# %%
hf_hub_download('gpt2', filename='config.json', local_dir = './tmp', local_dir_use_symlinks = 'False')
# %%
cache_info = scan_cache_dir()
# %%
snapshot_download('gpt2')
# %%
