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
```

