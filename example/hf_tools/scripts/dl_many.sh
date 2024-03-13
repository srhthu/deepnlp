hub_cach=/next_share/hf_cache/hub

python download_hf.py meta-llama/Llama-2-7b-hf --cache_dir $hub_cach
python download_hf.py meta-llama/Llama-2-7b-chat-hf --cache_dir $hub_cach
python download_hf.py meta-llama/Llama-2-13b-hf --cache_dir $hub_cach
python download_hf.py meta-llama/Llama-2-13b-chat-hf --cache_dir $hub_cach

python download_hf.py mistralai/Mistral-7B-v0.1 --cache_dir $hub_cach
python download_hf.py mistralai/Mistral-7B-Instruct-v0.2 --cache_dir $hub_cach