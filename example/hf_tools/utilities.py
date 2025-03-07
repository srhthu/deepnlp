from datetime import datetime
from huggingface_hub import HfApi, scan_cache_dir, try_to_load_from_cache
from pathlib import Path

def handle_info_value(x):
    """Convert info value type to int or string"""
    if isinstance(x, Path):
        return str(x)
    return x

def cvt_cache_info(cache_info):
    """Convert HF cache_info object to dict"""
    size_on_disk = cache_info.size_on_disk
    repos = [cvt_repo_info(k) for k in cache_info.repos]
    return {"size_on_disk": size_on_disk, "repos": repos}

def cvt_repo_info(repo_info):
    """Convert HF repo_info object to dict"""
    dt = {}
    for k,v in repo_info.__dict__.items():
        if k.startswith('_'):
            continue
        if k == 'refs':
            dt[k] = list(v)
            continue
        if k == 'revisions':
            dt[k] = [cvt_rev_info(e) for e in v]
            continue
        dt[k] = handle_info_value(v)
    return dt

def cvt_rev_info(rev_info):
    """Convert HF revision_info object to dict"""
    dt = {}
    for k,v in rev_info.__dict__.items():
        if k.startswith('_'):
            continue
        if k == 'files':
            dt[k] = len(v)
            continue
        if k == 'refs':
            dt[k] = list(v)
            continue
        dt[k] = handle_info_value(v)
    return dt