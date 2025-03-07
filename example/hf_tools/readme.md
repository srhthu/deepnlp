# Huggingface Hub Tools

This repository is to download model from huggingface and customize local caches.

Two environment variable:
- `HF_HOME`: cache dir for all huggingface_hub data
- `HF_HUB_CACHE`: default to '$HF_HOME/hub', only store HF repositories


## Change Permissions

Change file attribute to disable deletion of any files
```Bash
# for root
chattr -R +a /next_share/hf_cache/hub/*
# do not change the attribute of the hub directory, 
# as some temporary files may be created during downloading.

# Allow change attributes of the ref/main file
chattr -R -a /next_share/hf_cache/hub/*/refs/main
```

## Clear Unused Files

Motivation: a repository may save several copies of weights in different formats. However, the transformers package need only one piece of weights to build models.

If weights are downloaded by `snapshot_download`, the whole repository will be downlaoded.

If weights are downloaded when calling `from_pretrained` of `PreTrainedModel`, only used weights will be downloaded.