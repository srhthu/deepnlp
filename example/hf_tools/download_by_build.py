"""
Downlaod models from hugggingface hub by building the model and tokenizer.
"""
from argparse import ArgumentParser
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_LIST = [
    'google/t5-v1_1-base',
    'google/t5-v1_1-large',
    'google/t5-v1_1-xl',
    'google/t5-v1_1-xxl',
    'google/flan-t5-base',
    'google/flan-t5-large',
    'google/flan-t5-xl',
    'google/flan-t5-xxl',
]

def build_model_all(path, cache_dir):
    _ = AutoTokenizer.from_pretrained(path, cache_dir = cache_dir)
    _ = AutoModel.from_pretrained(path, torch_dtype = torch.bfloat16, device_map = 'cpu',
                                  cache_dir = cache_dir)

def download_all(cache_dir):
    for name in MODEL_LIST:
        print(f'Downloading {name}')
        build_model_all(name, cache_dir)
        print(f'Finished {name}')

def main():
    parser = ArgumentParser()
    parser.add_argument('repo_id', help = '`all` for download all models, or specify one model name to download')
    parser.add_argument('--cache_dir')
    args = parser.parse_args()

    if args.repo_id == 'all':
        download_all(args.cache_dir)
    else:
        build_model_all(args.repo_id, args.cache_dir)

if __name__ == '__main__':
    main()